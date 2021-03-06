import typing

import cv2
import numpy as np


def processing_slow(img: np.ndarray, shape: typing.Tuple[int, int] = (800, 800)) \
        -> np.ndarray:
    img = cv2.resize(img, shape)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img


def processing(img: np.ndarray, shape: typing.Tuple[int, int] = (800, 800), kernel_size: int = 3) \
        -> np.ndarray:
    h, w, *_ = img.shape
    if w < h:
        gap = (h - w) // 2
        img = img[gap:gap + w, :]
    elif h < w:
        gap = (w - h) // 2
        img = img[:, gap:gap + h]

    img = cv2.resize(img, shape)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    return img
