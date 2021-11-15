import argparse
from pathlib import Path
import multiprocessing as mp

import cv2


def extract_mov(args):
    mov_path, target_path, target_raw_path, skip, target_width, target_height = args
    print(f'start working {mov_path.stem}')

    video = cv2.VideoCapture(str(mov_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    skip = int(fps * skip + .5)

    idx = -1
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        idx += 1
        if idx < skip:
            continue

        if not (tar_name := target_raw_path.joinpath(f'{mov_path.stem}_{idx - skip:05d}.jpg')).exists():
            cv2.imwrite(str(tar_name), frame)
        
        if not (tar_name := target_path.joinpath(f'{mov_path.stem}_{idx - skip:05d}.jpg')).exists():
            processed = cv2.resize(frame, (target_width, target_height))
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
            cv2.imwrite(str(tar_name), processed)
    video.release()


def main(args: argparse.Namespace):
    src = Path(args.source)
    tar = Path(args.target)
    tar_raw = Path(args.target_raw)

    tar.mkdir(exist_ok=True, parents=True)
    tar_raw.mkdir(exist_ok=True, parents=True)

    worker = args.worker or mp.cpu_count()
    skip = args.skip
    target_width = args.target_width
    target_height = args.target_height

    arguments = [
        (mov, tar, tar_raw, skip, target_width, target_height)
        for mov in sorted(src.glob('*.mp4'))
    ]

    pool = mp.Pool(worker)
    pool.map(extract_mov, arguments)
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, default=0,
                        help="worker counts")

    parser.add_argument('--skip', type=float, default=60.,
                        help="skip first n seconds")
    parser.add_argument('--target-width', type=int, default=800)
    parser.add_argument('--target-height', type=int, default=800)

    parser.add_argument('--source', type=str, default='E:\\datasets\\fly\\meta\\mov')
    parser.add_argument('--target', type=str, default='E:\\datasets\\fly\\meta\\images')
    parser.add_argument('--target-raw', type=str, default='E:\\datasets\\fly\\meta\\images-raw')

    main(parser.parse_args())
