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


class Utils:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class DistanceMetrics:
    @staticmethod
    def euclidean_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        sum_squares_first = np.reshape(np.sum(first * first, axis=1), (first.shape[0], 1))
        sum_squares_second = np.reshape(np.sum(second * second, axis=1), (1, second.shape[0]))
        result = -2 * first @ second.T + sum_squares_second + sum_squares_first

        return np.sqrt(result)


class EvaluationMetrics:
    @staticmethod
    def rmse(x, y):
        return np.sqrt(np.mean((x - y) ** 2))
