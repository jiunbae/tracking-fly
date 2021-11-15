import argparse
from pathlib import Path
import json
import typing
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm


def main(args: argparse.Namespace):
    image_dir = Path(args.images)
    tracking_file = Path(args.tracking)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    with tracking_file.open() as f:
        tracking = json.load(f)

    np.random.seed(args.seed)
    colors = np.random.randint(0, 255, size=(len(tracking), 3), dtype=np.uint8)

    capture = cv2.VideoWriter(
        str(output_dir.joinpath(tracking_file.with_suffix('.mp4').name)),
        cv2.VideoWriter_fourcc(*'FMP4'),
        args.fps,
        (args.size, args.size),
    )
    for frame_id, image in enumerate(tqdm(sorted(image_dir.glob('*.jpg')))):
        img = cv2.imread(str(image))

        for track_id, track in enumerate(tracking):
            begin = track['start_frame']
            size = len(track['bbox'])
            if begin <= frame_id < begin + size:
                bbox = track['bbox'][frame_id - begin]
                cv2.rectangle(
                    img,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    tuple(map(int, colors[track_id])),
                    2,
                )

        capture.write(img)

        if args.end != 0 and frame_id > args.end:
            break

    capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='./data/images')
    parser.add_argument('--tracking', type=str, default='./data/tracking')

    parser.add_argument('--output', type=str, default='./data/videos')

    parser.add_argument('--end', type=int, default=0)

    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--size', type=int, default=800)
    parser.add_argument('--seed', type=int, default=42)

    main(parser.parse_args())
