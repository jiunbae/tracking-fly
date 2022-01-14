import typing
from pathlib import Path
from operator import itemgetter

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


def test_score(
    tracks: typing.List[typing.Dict[str, typing.Any]],
    image_path: Path,
    total: int = 0,
    step: int = 256,
) -> np.ndarray:
    def get_score(
        boxes: np.ndarray,
        images: typing.List[np.ndarray],
    ) -> float:

        acc = 0.
        try:
            filtered = boxes[::step]
            for (x1, y1, x2, y2), image in zip(filtered.astype(int), images):
                crop = image[y1:y2, x1:x2]
                acc += crop.mean()
        except Exception:
            acc += 9999999
        return acc

    images_all = [
        cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        for image_file in sorted(image_path.glob('*.jpg'))[::step]
    ]

    scores = np.array([
        -1 if len(track['bboxes']) < total else get_score(track['bboxes'], images_all)
        for track in tracks
    ])

    return scores


def refine(
    tracks: typing.List[typing.Dict[str, typing.Any]],
    image_path: Path,
    gt_count: int = 0,
    step: int = 1,
    total: int = 100,
) -> np.ndarray:
    total = max(total, *[len(track["bboxes"]) for track in tracks])
    scores = test_score(tracks, image_path, total=total)

    if gt_count != 0:
        if len(tracks) > gt_count:
            idx = scores.argsort()[:gt_count]
            tracks = [
                track
                for i, track in enumerate(tracks)
                if i in idx
            ]

            Warning(f"{len(tracks)} {gt_count} mismatch")

    for track in tracks:
        if len(track['bboxes']) != total:
            first_frame = track['bboxes'][0]
            track['bboxes'] = np.concatenate((
                np.repeat(first_frame, total - len(track['bboxes']), axis=0).reshape(-1, 4),
                track['bboxes']
            ))

    track_boxes = tuple(map(itemgetter('bboxes'), tracks))

    return np.stack(track_boxes)[:, ::step, :]
