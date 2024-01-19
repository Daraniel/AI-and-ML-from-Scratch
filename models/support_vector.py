from numbers import Number

import numpy as np

from common.base_model import BaseModel, Classifier
from common.exceptions import (InvalidArgumentException,
                               ModelNotTrainedException)


class SVM(BaseModel, Classifier):
    def __init__(self, lambda_regularization: float = 1):
        """
        creates a support vector machine model
        :param lambda_regularization: lambda regularization term
        """
        self.weights = None
        if not isinstance(lambda_regularization, Number):
            raise InvalidArgumentException("lambda_regularization must be a number")
        if lambda_regularization <= 0:
            raise InvalidArgumentException(
                "lambda_regularization must be a positive number"
            )
        self.lambda_regularization = lambda_regularization

    def learn(
        self, list_of_features: np.array, list_of_targets: np.array, epochs: int = 100
    ):
        """
        trains the model using a list of input features to predict the list of targets
        :param list_of_features: list of input features
        :param list_of_targets: list of labels
        :param epochs: number of epochs to train the model
        """
        if len(list_of_features) != len(list_of_targets):
            raise InvalidArgumentException(
                "length of features and targets must be same"
            )
        if not isinstance(epochs, int):
            raise InvalidArgumentException("Number of epochs must be an integer")
        if epochs <= 0:
            raise InvalidArgumentException("Number of epochs must be a positive number")

        self.weights = [0] * (len(list_of_features) + 1)

        last = 0
        for i, (features, target) in enumerate(zip(list_of_features, list_of_targets)):
            if target == last:
                continue

            alpha = 1 / (self.lambda_regularization * (i + 1))
            self._learn(features, target, alpha)
            last = target

    def _learn(self, features, target, alpha):
        if target * self.infer(features) < 1:
            for i, feature in enumerate(features):
                self.weights[i] = self.weights[i] + alpha * (
                    (target * feature)
                    + (-2 * self.lambda_regularization * self.weights[i])
                )

        else:
            for i in range(len(features)):
                self.weights[i] = self.weights[i] + alpha * (
                    -2 * self.lambda_regularization * self.weights[i]
                )

    def infer(self, features: np.array) -> np.array:
        """
        predicts the labels of a single input
        :param features: input features
        :return: predicted label of the input
        """
        if self.weights is None:
            raise ModelNotTrainedException("Please train the model before using it!")

        result = 0
        for i, feature in enumerate(features):
            result += self.weights[i] * feature
        return result

    def infer_inputs(self, inputs_features: np.array) -> np.array:
        """
        predicts the labels of inputs
        :param inputs_features: features of inputs
        :return: predicted label of inputs
        """
        return np.array([self.infer(features) for features in inputs_features])


# class SVR(BaseModel, Regression):
#     def __init__(self, epsilon=0.5):
#         self.epsilon = epsilon
#
#     def learn(self, features: np.array, targets: np.array, epochs=100, learning_rate=0.1):
#         pass
#
#     def infer(self, features: np.array) -> np.array:
#         pass
