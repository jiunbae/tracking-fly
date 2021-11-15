import argparse
import json
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
    gt_file = Path(args.gt)

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

    progress = tqdm(videos := sorted(video_dir.glob('*.mp4')))
    for video in progress:
        progress.set_description(f'{video.stem}')
        gt_count = gt.get(video.name, 0)

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
        skip = int(args.skip * fps)

        capture.set(cv2.CAP_PROP_POS_FRAMES, skip)

        tracker = Tracker(gt_count=gt_count, center=(args.height // 2, args.width // 2))
        video_progress = tqdm(range((frame_count - skip) // args.batch * args.step))
        while frame_index + skip < frame_count:
            ret, frame = capture.read()

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
                    batch_index = frame_index - args.batch + result_idx

                    result = filtering.processing(batch_inputs[result_idx - 1], result)
                    tracker.update(result)

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
                        path=str(dump_detection_dir.joinpath(f'{batch_index:06d}.csv')),
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
        capture.release()
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

        refined_tracks = filtering.refine(tracks, gt_count=gt_count, step=args.refine_step)
        analysis = Analysis(
            refined_tracks,
            base_dir=dump_analysis_dir,
            cluster_distance_threshold=args.cluster_distance_threshold,
            cluster_time_threshold=fps * args.cluster_time_threshold,
            width=args.width,
            height=args.height,
        )
        analysis.draw_all()
        analysis.df.to_csv(dump_analysis_dir.joinpath('data.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data',
                        help="Path to dataset directory")
    parser.add_argument('--skip', type=float, default=60.,
                        help="Skip first {skip} frames")

    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=800)

    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--refine-step', type=int, default=4)

    parser.add_argument('--cluster-distance-threshold', type=float, default=300)
    parser.add_argument('--cluster-time-threshold', type=float, default=30.)

    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--weights', type=str, default='weights/efficientdet-d4.pth')
    parser.add_argument('--compound-coefficient', type=int, default=4)
    parser.add_argument('--gt', type=str, default='./data/video/gt_count.json')

    parser.add_argument('--no-dump', action='store_true', default=False,
                        help="Don't dump images")

    main(parser.parse_args())
