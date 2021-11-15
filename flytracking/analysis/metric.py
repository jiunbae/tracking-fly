import numpy as np


def center(box: np.ndarray) \
        -> np.ndarray:
    return np.stack((
        (box[..., 2] - box[..., 0]) / 2 + box[..., 0],
        (box[..., 3] - box[..., 1]) / 2 + box[..., 1],
    ), axis=-1)

