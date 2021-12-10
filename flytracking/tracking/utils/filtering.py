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
        image[int(y1):int(y2), int(x1):int(x2)].std() < 30 and acc < .5
        for x1, y1, x2, y2, acc in boxes
    ]

    size_filter = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 5000

    return size_filter | crops


def refine(
    tracks: typing.List[typing.Dict[str, typing.Any]],
    gt_count: int = 0,
    step: int = 1,
    total: int = 100,
) -> np.ndarray:
    if gt_count != 0:
        if len(tracks) != gt_count:
            Warning(f"{len(tracks)} {gt_count} mismatch")

    total = max(total, *[len(track["bboxes"]) for track in tracks])

    for track in tracks:
        if len(track['bboxes']) != total:
            first_frame = track['bboxes'][0]
            track['bboxes'] = np.concatenate((
                np.repeat(first_frame, total - len(track['bboxes']), axis=0).reshape(-1, 4),
                track['bboxes']
            ))

    track_boxes = tuple(map(itemgetter('bboxes'), tracks))

    return np.stack(track_boxes)[:, ::step, :]
