import numpy as np

from common.exceptions import ModelNotTrainedException


class AverageNormalizer2D:
    def __init__(self):
        self.average = None
        self.std = None

    def calculate(self, x: np.ndarray):
        self.average = np.mean(x, axis=0)
        self.std = np.std(x)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.average is None:
            raise ModelNotTrainedException()

        return (x - self.average) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        if self.average is None:
            raise ModelNotTrainedException()

        return x * self.std + self.average

