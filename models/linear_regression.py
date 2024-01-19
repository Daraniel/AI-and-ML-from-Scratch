import random

import numpy as np

from common.base_model import BaseModel, Regression
from common.exceptions import (InvalidArgumentException,
                               ModelNotTrainedException)
from common.utils import AverageNormalizer2D


class LinearRegressionAnalytic(BaseModel, Regression):
    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_normalizer = AverageNormalizer2D()
        self.target_normalizer = AverageNormalizer2D()

    @staticmethod
    def _calculate_weights(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        calculates the weights using the following equation:
        (X^T . X)^(-1) . X^T . Y^T
        Where X is the normalized value of features (value - average_value) and Y is the normalized value of targets
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, 1) (with N being the number of samples as in the provided features and
         1 being the number of target dimensions)
        :return: numpy array of shape (d) (with d being the number of feature dimensions) that maps features to targets
        """
        mid_result = features.T.dot(features)

        try:
            mid_result = np.linalg.inv(mid_result)
        except np.linalg.LinAlgError:
            print(
                "Warning, normal matrix inverse failed, using  pseudo-inverse instead."
            )
            mid_result = np.linalg.pinv(mid_result)
        return mid_result.dot(features.T).dot(targets).flatten()

    @staticmethod
    def _calculate_bias(
        features: np.ndarray, targets: np.ndarray, weights: np.ndarray
    ) -> float:
        """
        calculates the bias using the following equation:
        bias = average_value_of_y - foreach_dimension_i_of_x: (average_of_i * weight_of_i)
        => bias = average_of_y - averages_of_x * weights_of_x
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, 1) (with N being the number of samples as in the provided features and
         1 being the number of target dimensions)
        :param weights: numpy array of shape (d) (with d being the number of feature dimensions)
        :return: float
        """
        # x_bar = np.mean(features, axis=0)
        # y_bar = np.mean(targets, axis=0)
        return float(targets - np.sum(features * weights))

    def learn(self, features: np.array, targets: np.array):
        """
        train the model
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, 1) (with N being the number of samples as in the provided features and
         1 being the number of target dimensions)
        :return:
        """
        if features.shape[0] != targets.shape[0]:
            raise InvalidArgumentException(
                "Features and targets must have similar first shape"
            )

        self.feature_normalizer.calculate(features)
        self.target_normalizer.calculate(targets)

        normalized_features = self.feature_normalizer.normalize(features)
        normalized_targets = self.target_normalizer.normalize(targets)

        average_of_features = np.mean(normalized_features, axis=0)
        average_of_targets = np.mean(normalized_targets, axis=0)

        self.weights = self._calculate_weights(normalized_features, normalized_targets)
        self.bias = self._calculate_bias(
            average_of_features, average_of_targets, self.weights
        )

    def infer(self, features: np.array):
        """
        predict the output using the weights and features
        :param features: np array of shape (N, d)
        :return: inferred value
        :raises: ModelNotTrainedException
        """
        features = self.feature_normalizer.normalize(features)
        if self.weights is None or self.bias is None:
            raise ModelNotTrainedException()
        output = np.sum(features * self.weights, axis=1) + self.bias
        # return output
        return self.target_normalizer.denormalize(output)


class LinearRegressionGradiantDecent(BaseModel, Regression):
    def __init__(self, batch_divider: int = 50):
        self.weights = None
        self.bias = None
        self.feature_normalizer = AverageNormalizer2D()
        self.target_normalizer = AverageNormalizer2D()
        self.batch_divider = batch_divider

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        array = array - np.mean(array, axis=0)
        return array

    @staticmethod
    def _step(
        features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        learning_rate: float,
        batch_divider: int,
    ) -> np.ndarray:
        """
        calculates the weights in the next step using gradiant decent method using the following formula:
        W_new = W_old - learning_rate * -X^T . (Y - X . W_old)
        Where X is the normalized value of features (value - average_value) and Y is the normalized value of targets and
        W is the weights
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, 1) (with N being the number of samples as in the provided features and
         1 being the number of target dimensions)
        :param weights: numpy array of shape (d) (with d being the number of feature dimensions)
        :param learning_rate: learning rate
        :param batch_divider: number of parts to divide the training dateset
        :return: numpy array of shape (d) (with d being the number of feature dimensions)
        """
        indices = list(range(features.shape[0]))
        for i in range(batch_divider):
            ix = random.sample(
                indices, min(features.shape[0] // batch_divider, len(indices))
            )
            for x in ix:
                indices.remove(x)
            gradiant = learning_rate * -features[ix].T.dot(
                targets[ix].flatten() - features[ix].dot(weights)
            )
            gradiant[gradiant < 0.0000000001] = 0.0000000001  # prevent vanishing values
            gradiant[gradiant > 1000000000] = 1000000000  # prevent exploding values
            weights = weights - gradiant
        return weights

    @staticmethod
    def _calculate_bias(
        features: np.ndarray, targets: np.ndarray, weights: np.ndarray
    ) -> float:
        """
        calculates the bias using the following equation:
        bias = average_value_of_y - foreach_dimension_i_of_x: (average_of_i * weight_of_i)
        => bias = average_of_y - averages_of_x * weights_of_x
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, 1) (with N being the number of samples as in the provided features and
         1 being the number of target dimensions)
        :param weights: numpy array of shape (d) (with d being the number of feature dimensions)
        :return: float
        """

        return float(targets - np.sum(features * weights))

    def learn(
        self,
        features: np.array,
        targets: np.array,
        learning_rate: float = 1,
        iterations: int = 10,
    ):
        """
        train the model
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, 1) (with N being the number of samples as in the provided features and
         1 being the number of target dimensions)
        :param iterations:
        :param learning_rate:
        :return:
        :raises: InvalidArgumentException
        """
        if learning_rate <= 0:
            raise InvalidArgumentException("Learning rate must be positive number")
        if not isinstance(iterations, int) or iterations < 1:
            raise InvalidArgumentException("Iterations must be a positive integer")
        if features.shape[0] != targets.shape[0]:
            raise InvalidArgumentException(
                "Features and targets must have similar first shape"
            )

        self.feature_normalizer.calculate(features)
        self.target_normalizer.calculate(targets)

        normalized_features = self.feature_normalizer.normalize(features)
        normalized_targets = self.target_normalizer.normalize(targets)

        average_of_features = np.mean(normalized_features, axis=0)
        average_of_targets = np.mean(normalized_targets, axis=0)

        self.weights = np.random.randn(features.shape[1])
        self.bias = 0

        for i in range(iterations):
            self.weights = self._step(
                normalized_features,
                normalized_targets,
                self.weights,
                learning_rate,
                self.batch_divider,
            )

        self.bias = self._calculate_bias(
            average_of_features, average_of_targets, self.weights
        )

    def infer(self, features: np.array) -> np.array:
        """
        predict the output using the weights and features
        :param features: np array of shape (N, d)
        :return: inferred value
        :raises: ModelNotTrainedException
        """
        features = self.feature_normalizer.normalize(features)
        if self.weights is None or self.bias is None:
            raise ModelNotTrainedException()
        output = np.sum(features * self.weights, axis=1) + self.bias
        return self.target_normalizer.denormalize(output)
