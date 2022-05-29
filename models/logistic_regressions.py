import random

import numpy as np

from common.base_model import BaseModel, Classifier
from common.exceptions import InvalidArgumentException, ModelNotTrainedException
from common.utils import AverageNormalizer2D, Utils


class LogisticRegression(BaseModel, Classifier):
    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_normalizer = AverageNormalizer2D()
        self.target_normalizer = AverageNormalizer2D()

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        array = array - np.mean(array, axis=0)
        return array

    @staticmethod
    def _step(features: np.ndarray, targets: np.ndarray, weights: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        calculates the weights in the next step using sigmoid gradiant decent method using the following formula:
        W_new = W_old - learning_rate * -X^T . (Y - sigmoid(X . W_old))
        Where X is the normalized value of features (value - average_value) and Y is the normalized value of targets and
        W is the weights
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, 1) (with N being the number of samples as in the provided features and
         1 being the number of target dimensions)
        :param weights: numpy array of shape (d) (with d being the number of feature dimensions)
        :param learning_rate: learning rate
        :return: numpy array of shape (d) (with d being the number of feature dimensions)
        """
        batch_divider = 50
        indices = list(range(features.shape[0]))
        for i in range(batch_divider):
            ix = random.sample(indices, min(features.shape[0]//batch_divider, len(indices)))
            for x in ix:
                indices.remove(x)
            gradiant = learning_rate * - features[ix].T.dot(targets[ix].flatten() - Utils.sigmoid(features[ix].dot(weights)))
            # gradiant[gradiant < 0.00000000001] = 0.00000000001  # prevent vanishing values
            # gradiant[gradiant > 10000000000] = 10000000000  # prevent exploding values
            weights = weights - gradiant
        return weights

        # return weights - learning_rate * (np.dot(features.T, (Utils.sigmoid(np.dot(features, weights))) - targets))

    @staticmethod
    def _calculate_bias(features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
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

    def learn(self, features: np.array, targets: np.array, learning_rate: float = 1, iterations: int = 10):
        """
        train the model
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, x) (with N being the number of samples as in the provided features and
         x being the number of target classes)
        :param iterations:
        :param learning_rate:
        :return:
        :raises: InvalidArgumentException
        """
        if learning_rate <= 0:
            raise InvalidArgumentException('Learning rate must be positive number')
        if not isinstance(iterations, int) or iterations < 1:
            raise InvalidArgumentException('Iterations must be a positive integer')
        if features.shape[0] != targets.shape[0]:
            raise InvalidArgumentException('Features and targets must have similar first shape')

        # self.feature_normalizer.calculate(features)
        # self.target_normalizer.calculate(targets)

        # normalized_features = self.feature_normalizer.normalize(features)
        # normalized_targets = self.target_normalizer.normalize(targets)

        # average_of_features = np.mean(features, axis=0)
        # average_of_targets = np.mean(targets, axis=0)

        # average_of_features = np.mean(normalized_features, axis=0)
        # average_of_targets = np.mean(normalized_targets, axis=0)

        # self.weights = np.random.randn(features.shape[1])
        self.weights = np.ones(features.shape[1])
        self.bias = 0

        for i in range(iterations):
            # self.weights = self._step(normalized_features, normalized_targets, self.weights, learning_rate)
            self.weights = self._step(features, targets, self.weights, learning_rate)

        # self.bias = self._calculate_bias(average_of_features, average_of_targets, self.weights)

    def infer(self, features: np.array) -> np.array:
        """
        predict the output using the weights and features
        :param features: np array of shape (N, d)
        :return: inferred values
        :raises: ModelNotTrainedException
        """
        # features = self.feature_normalizer.normalize(features)
        if self.weights is None or self.bias is None:
            raise ModelNotTrainedException()
        output = np.sum(features * self.weights, axis=1) + self.bias
        return np.round(output)
        # return self.target_normalizer.denormalize(output)
