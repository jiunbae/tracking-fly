import numpy as np


class BatchLoader:
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.items = []

    @property
    def full(self) \
            -> bool:
        return len(self.items) >= self.batch_size

    @property
    def empty(self) \
            -> bool:
        return len(self.items) == 0

    def push(self, item: np.ndarray):
        self.items.append(item)

    def pop(self) \
            -> np.ndarray:
        results = np.stack(self.items)
        self.items = []
        return results
