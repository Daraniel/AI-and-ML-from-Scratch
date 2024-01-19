import abc

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


class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def softmax(x, axis=None):
        x_max = np.amax(x, axis=axis, keepdims=True)
        exp_x_shifted = np.exp(x - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


class ActivationFunctionsForNN:
    class BaseActivationFunctionForNN(abc.ABC):
        @abc.abstractmethod
        def forward(self, x):
            pass

        def backward(self, x):
            pass

    class Relu(BaseActivationFunctionForNN):
        def __init__(self):
            pass

        def forward(self, x):
            return ActivationFunctions.relu(x)

        def backward(self, x):
            return 1.0 * (x > 0)

    class Sigmoid(BaseActivationFunctionForNN):
        def __init__(self, shape):
            self.result = np.zeros(shape)

        def forward(self, x):
            self.result = ActivationFunctions.sigmoid(x)
            return self.result

        def backward(self, x):
            return x * self.result * (1 - self.result)

    class Softmax(BaseActivationFunctionForNN):
        def __init__(self, axis=None):
            self.axis = axis

        def forward(self, x):
            return ActivationFunctions.softmax(x, self.axis)

        def backward(self, x):
            return x


class EvaluationMetrics:
    @staticmethod
    def rmse(x, y):
        return np.sqrt(np.mean((x - y) ** 2))

    @staticmethod
    def hinge_lost(x, y):
        return np.maximum(0, 1 - x * y)

    @staticmethod
    def cross_entropy_loss(yhat, y):
        yhat = np.clip(yhat, 1e-10, 1 - 1e-10)
        aa = y * np.log(yhat)
        return -np.nansum(aa)

    @staticmethod
    def euclidean_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        sum_squares_first = np.reshape(
            np.sum(first * first, axis=1), (first.shape[0], 1)
        )
        sum_squares_second = np.reshape(
            np.sum(second * second, axis=1), (1, second.shape[0])
        )
        result = -2 * first @ second.T + sum_squares_second + sum_squares_first
        return np.sqrt(result)


class ImpurityFunctions:
    @staticmethod
    def gini(x):
        return 1 - np.sum(x**2)

    @staticmethod
    def entropy(x):
        x[np.where(x == 0)] = 1
        return -np.sum(x * np.log2(x))

    @staticmethod
    def mis_classification(x):
        return 1 - np.max(x)
