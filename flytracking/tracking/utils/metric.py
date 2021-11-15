import typing
from functools import partial

import numpy as np


BOX = typing.Union[
    np.ndarray,
    typing.List[typing.Union[int, float]],
    typing.Tuple[typing.Union[int, float], ...],
]


def center(bbox: BOX) \
        -> typing.Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1


def distance(bbox1: BOX, bbox2: BOX) \
        -> float:
    (x1, y1), (x2, y2) = center(bbox1), center(bbox2)
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5


def size(bbox: BOX) \
        -> float:
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) * (y2 - y1)) ** .5


def blend(bbox1: BOX, bbox2: BOX, ratio: float) \
        -> BOX:
    def blend_(x1_: float, x2_: float, ratio_: float) \
            -> float:
        return (x2_ - x1_) * ratio_ + x1_

    x11, y11, x12, y12 = bbox1
    w, h = x12 - x11, y12 - y11
    (x1, y1), (x2, y2) = center(bbox1), center(bbox2)
    x, y = blend_(x1, x2, ratio), blend_(y1, y2, ratio)

    return list(map(int, (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))


def iou(bbox1: BOX, bbox2: BOX) \
        -> float:
    (x0_1, y0_1, x1_1, y1_1) = map(float, bbox1)
    (x0_2, y0_2, x1_2, y1_2) = map(float, bbox2)

    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def get_best_match(bbox: BOX, bboxes: typing.Union[np.ndarray, BOX]) \
        -> typing.Tuple[int, float, BOX]:
    iou_array = np.array(list(map(
        partial(iou, bbox),
        bboxes,
    )))

    best_idx = iou_array.argmax()
    best_match = bboxes[best_idx]
    return best_idx, iou_array[best_idx], best_match


def get_unique(iou_map: np.ndarray, axis: int = -1) \
        -> bool:
    argmax = iou_map.argmax(axis=axis)
    unique = np.unique(argmax)

    if len(unique) != len(argmax):
        return False

    return True
