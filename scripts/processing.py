import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def processing(img):
    img = cv2.resize(img, (720, 720))
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img


def main(args: argparse.Namespace):
    src = Path(args.source)
    tar = Path(args.target)
    prefix = args.prefix

    for image in tqdm(sorted(src.glob(f'{prefix}.jpg'))):
        img = cv2.imread(str(image))
        img = processing(img)
        cv2.imwrite(str(tar.joinpath(image.name)), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='*')
    parser.add_argument('--source', type=str, default='E:\\datasets\\fly\\images-raw')
    parser.add_argument('--target', type=str, default='E:\\datasets\\fly\\images')

    main(parser.parse_args())
