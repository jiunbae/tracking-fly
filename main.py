import argparse
import json
import math
from pathlib import Path
from operator import itemgetter

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

from flytracking.detection import Detector
from flytracking.detection.utils import preprocess
from flytracking.tracking import Tracker
from flytracking.tracking.utils import filtering
from flytracking.utils import Callback, BatchLoader
from flytracking.analysis import Analysis, vis


def main(args: argparse.Namespace):
    data_dir = Path(args.data)
    video_dir = data_dir.joinpath('video')
    gt_file = data_dir.joinpath(Path(args.gt))

    dump_image_base_dir = data_dir.joinpath('images')
    dump_image_base_dir.mkdir(exist_ok=True, parents=True)
    dump_detection_base_dir = data_dir.joinpath('detections')
    dump_detection_base_dir.mkdir(exist_ok=True, parents=True)
    dump_tracking_base_dir = data_dir.joinpath('tracking')
    dump_tracking_base_dir.mkdir(exist_ok=True, parents=True)
    dump_analysis_base_dir = data_dir.joinpath('analysis')
    dump_analysis_base_dir.mkdir(exist_ok=True, parents=True)

    callback = Callback(trigger=(not args.no_dump))
    detector = Detector(
        weights=args.weights,
        compound_coef=args.compound_coefficient,
    )
    dataloader = BatchLoader(args.batch)

    gt = {}
    if gt_file.exists():
        with gt_file.open() as f:
            gt = json.load(f)

    progress = tqdm(videos := sorted(video_dir.glob(f'*{args.ext}')))
    for video in progress:
        if video.stem != 'KakaoTalk_20211124_171806444':
            continue

        progress.set_description(f'{video.stem}')
        gt_count = gt.get(video.stem, 0)

        # define paths
        dump_image_dir = dump_image_base_dir.joinpath(video.stem)
        dump_image_dir.mkdir(exist_ok=True, parents=True)
        dump_detection_dir = dump_detection_base_dir.joinpath(video.stem)
        dump_detection_dir.mkdir(exist_ok=True, parents=True)
        dump_analysis_dir = dump_analysis_base_dir.joinpath(video.stem)
        dump_analysis_dir.mkdir(exist_ok=True, parents=True)

        capture = cv2.VideoCapture(str(video))

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_index, frame_count = 0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = math.ceil(args.skip * fps)

        if args.no_overwrite and (result_path := dump_tracking_base_dir.joinpath(f'{video.stem}.json')).exists():
            with result_path.open() as f:
                tracks = json.load(f)
            tracks = [
                {
                    'bboxes': np.array(track['bbox']),
                    'start_frame': track['start_frame'],
                    'lost': track['lost'],
                } for track in tracks
            ]
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, skip)

            tracker = Tracker(
                gt_count=gt_count,
                center=(args.height // 2, args.width // 2),
                init_idx=args.init_frame,
            )
            video_progress = tqdm(range(frame_count - skip))
            while frame_index + skip < frame_count:
                *_, frame = capture.read()

                if not (frame_index % args.step):
                    inputs = preprocess.processing(frame, shape=(args.height, args.width))

                    dataloader.push(inputs)
                    callback.write_image(
                        path=str(dump_image_dir.joinpath(f'{frame_index:06d}.jpg')),
                        image=inputs,
                    )

                if dataloader.full:
                    batch_inputs = dataloader.pop()
                    results = detector(batch_inputs)

                    for result_idx, result in enumerate(results, start=1):
                        batch_index = frame_index // args.step - args.batch + result_idx

                        processed_result = filtering.processing(batch_inputs[result_idx - 1], result)
                        tracker.update(processed_result)

                        if batch_index == 0:
                            batch_input, *_ = batch_inputs

                            for color, points in zip(vis.colors(), map(
                                itemgetter(-1),
                                map(
                                    itemgetter('bboxes'),
                                    tracker.tracks_active,
                                ),
                            )):
                                x1, y1, x2, y2 = tuple(map(int, points))
                                cv2.rectangle(batch_input, (x1, y1), (x2, y2), color, 3)
                            callback.write_image(
                                path=str(dump_image_base_dir.joinpath(f'{video.stem}.jpg')),
                                image=batch_input,
                            )

                        callback.write_csv(
                            path=str(dump_detection_dir.joinpath(f'{frame_index:06d}.csv')),
                            data_frame=pd.DataFrame(
                                result,
                                columns=['x1', 'y1', 'x2', 'y2', 'acc'],
                            ),
                            index=False,
                        )

                video_progress.update()
                frame_index += 1

            if not dataloader.empty:
                batch_inputs = dataloader.pop()
                results = detector(batch_inputs)

                for result_idx, result in enumerate(results, start=1):
                    callback.write_csv(
                        path=str(dump_detection_dir.joinpath(
                            f'{frame_index - args.batch + result_idx:06d}.csv'
                        )),
                        data_frame=pd.DataFrame(
                            result,
                            columns=['x1', 'y1', 'x2', 'y2', 'acc'],
                        ),
                        index=False,
                    )

                    result = filtering.processing(batch_inputs[result_idx - 1], result)
                    tracker.update(result)

            video_progress.close()
            tracks = tracker.release()
            callback.write_json(
                path=str(dump_tracking_base_dir.joinpath(f'{video.stem}.json')),
                data=[
                    {
                        'bbox': np.stack(track['bboxes']).astype(int).tolist(),
                        'start_frame': track['start_frame'],
                        'lost': track['lost'],
                    } for track in tracks
                ],
                indent=4,
            )
        capture.release()

        refined_tracks = filtering.refine(
            tracks,
            gt_count=gt_count,
            step=args.refine_step,
            total=round((frame_count - skip) / args.step),
        )
        
        analysis = Analysis(
            refined_tracks,
            base_dir=dump_analysis_dir,
            cluster_distance_threshold=args.cluster_distance_threshold,
            cluster_time_threshold=int((fps * args.cluster_time_threshold) / (args.step * args.refine_step)),
            cluster_count_threshold=args.cluster_count_threshold,
            interaction_step=args.interaction_step,
            interaction_distance_threshold=args.interaction_distance_threshold,
            analysis_best_count=args.analysis_best_count,

            color_density=args.color_density,

            width=args.width,
            height=args.height,
            skip=skip,
            step=args.step * args.refine_step,
            fps=fps,
        )
        analysis.draw_tracks()
        analysis.dump_cluster()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data',
                        help="Path to dataset directory")
    parser.add_argument('--ext', type=str, default='.mp4')
    parser.add_argument('--skip', type=float, default=60.,
                        help="Skip first {skip} frames (seconds)")

    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=800)

    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--refine-step', type=int, default=4)

    parser.add_argument('--cluster-distance-threshold', type=float, default=285,
                        help="cluster minimum distance (800 is 100%, default=285)")
    parser.add_argument('--cluster-time-threshold', type=float, default=10.,
                        help="cluster minimum time (seconds)")
    parser.add_argument('--cluster-outlier-threshold', type=int, default=5,
                        help="cluster outlier max count")
    parser.add_argument('--interaction-step', type=float, default=3.,
                        help="interaction step (seconds)")
    parser.add_argument('--interaction-distance-threshold', type=float, default=100,
                        help="interaction distance threshold")
    parser.add_argument('--analysis-best-count', type=int, default=3,
                        help="distance report best n-th")

    parser.add_argument('--color-density', type=str, default='Reds', choices=[
        'Reds', 'Blues', 'Jet',
        'Oranges', 'Purples', 'Greens',
        'Greys', 'BuGn', 'BuPu',
        'GnBu', 'OrRd', 'PuBu',
        'PuBuGn', 'PuRd', 'RdPu',
        'YlGn', 'YlGnBu', 'YlOrBr',
        'YlOrRd', 'Spectral', 'RdBu',
        'RdGy', 'RdYlBu', 'RdYlGn',
    ], help="density plot color")

    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--weights', type=str, default='weights/efficientdet-d4.pth')
    parser.add_argument('--compound-coefficient', type=int, default=4)
    parser.add_argument('--gt', type=str, default='video/gt_count.json')

    parser.add_argument('--init-frame', type=int, default=100,
                        help="Tracker initialize frame")

    parser.add_argument('--no-dump', action='store_true', default=False,
                        help="Don't dump images")
    parser.add_argument('--no-overwrite', action='store_true', default=False,
                        help="Use exist tracking results")

    main(parser.parse_args())
