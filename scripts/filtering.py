import argparse
from pathlib import Path
import multiprocessing as mp

import cv2
import pandas as pd

def is_bg(crop, box, acc):
    x1, y1, x2, y2 = box
    
    return (x2 - x1) * (y2 - y1) > 5000 or (acc < .5 and crop.std() < 30) 


def filtered(args):
    image_dir, annot_dir, output_dir, bg_dir = args
    print(f'starting {annot_dir.stem}')
    output_dir.mkdir(exist_ok=True, parents=True)
    bg_dir.mkdir(exist_ok=True, parents=True)
    images = sorted(image_dir.glob('*.jpg'))
    annots = sorted(annot_dir.glob('*.csv'))

    for image, annot in zip(images, annots):
        assert image.stem == annot.stem, f"{image.stem} is not matched {annot.stem}"

        img = cv2.imread(str(image), 0)
        df = pd.read_csv(str(annot), header=None)
    
        bgs = []
        for idx, (*points, acc) in enumerate(df.values):
            x1, y1, x2, y2 = map(int, points)
            
            crop = img[y1:y2, x1:x2]
            bg = is_bg(crop, (x1, y1, x2, y2), acc)
            bgs.append(int(bg))

            if bg:
                cv2.imwrite(str(bg_dir.joinpath(f'{annot.stem}_{idx:03d}.jpg')), crop)

        df['bg'] = bgs
        df.to_csv(str(output_dir.joinpath(annot.name)), header=None, index=None)


def main(args: argparse.Namespace):
    image_dir = Path(args.images)
    annot_dir = Path(args.annotations)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    bg_dir = Path(args.test)
    bg_dir.mkdir(exist_ok=True, parents=True)
    worker = args.worker or mp.cpu_count()

    image_dirs = sorted(image_dir.iterdir())
    annot_dirs = sorted(annot_dir.iterdir())

    arguments = [
        (image, annot, output_dir.joinpath(annot.stem), bg_dir.joinpath(annot.stem))
        for image, annot in zip(image_dirs, annot_dirs)
        if image.stem == "20210924_094923"
    ]
    
    pool = mp.Pool(worker)
    pool.map(filtered, arguments)
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, default=0,
                        help="worker counts")

    parser.add_argument('--images', type=str, default='E:\\datasets\\fly\\meta\\images')
    parser.add_argument('--annotations', type=str, default='E:\\datasets\\fly\\meta\\annotations')
    parser.add_argument('--output', type=str, default='E:\\datasets\\fly\\meta\\annotations-filtered')
    parser.add_argument('--test', type=str, default='E:\\datasets\\fly\\meta\\bg-case')

    main(parser.parse_args())
