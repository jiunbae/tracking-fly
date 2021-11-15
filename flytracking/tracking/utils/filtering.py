import typing
from operator import itemgetter
from itertools import groupby

import cv2
import numpy as np


def processing(image: np.ndarray, boxes: np.ndarray) \
        -> np.ndarray:
    backgrounds = is_bg(image, boxes)
    result = boxes[np.where(~backgrounds)]
    return result


def is_bg(image: np.ndarray, boxes: np.ndarray) \
        -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    crops = [
        image[int(y1):int(y2), int(x1):int(x2)].std() < 33
        for x1, y1, x2, y2, acc in boxes
    ]

    size_filter = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 5000

    return size_filter | crops


def refine(
    tracks: typing.List[typing.Dict[str, typing.Any]],
    gt_count: int = 0,
    step: int = 1,
) -> np.ndarray:
    if gt_count != 0:
        assert len(tracks) == gt_count
    tracks = tuple(map(itemgetter('bboxes'), tracks))

    assert all(len(track) == len(tracks[0]) for track in tracks)

    return np.stack(tracks)[:, ::step, :]
