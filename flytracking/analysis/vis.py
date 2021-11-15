import typing

import numpy as np


def colors(size: int = 128, seed: int = 0) \
        -> typing.List[typing.Tuple[int, ...]]:
    np.random.seed(seed)
    return [
        tuple(map(int, color))
        for color in np.random.randint(0, 255, size=(size, 3))
    ]
