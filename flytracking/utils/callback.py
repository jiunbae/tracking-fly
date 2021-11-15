import typing
import json

import cv2
import numpy as np
import pandas as pd


class Callback:
    def __init__(self, trigger: bool = True):
        self.trigger = trigger

    def write_image(self, path: str, image: np.ndarray):
        if self.trigger:
            cv2.imwrite(path, image)

    def write_csv(self, path: str, data_frame: pd.DataFrame, *args, **kwargs):
        if self.trigger:
            data_frame.to_csv(path, *args, **kwargs)

    def write_json(self, path: str, data: typing.Any, *args, **kwargs):
        if self.trigger:
            with open(path, 'w') as f:
                json.dump(data, f, *args, **kwargs)
