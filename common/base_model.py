import abc

import numpy as np


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def learn(self, features: np.array, targets: np.array):
        """
        train the model
        :param features: numpy array of shape (N, d) (with N being the number of samples and d being the number of
         feature dimensions)
        :param targets: numpy array of shape (N, 1) (with N being the number of samples as in the provided features and
         1 being the number of target dimensions)
        :return:
        """
        pass

    @abc.abstractmethod
    def infer(self, features: np.array) -> np.array:
        """
        predict the output using the weights and features
        :param features: np array of shape (N, d)
        :return: inferred value
        :raises: ModelNotTrainedException
        """
        pass


class Classifier:
    pass


class Regression:
    pass
