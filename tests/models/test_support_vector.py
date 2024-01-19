import numpy as np
import pytest

from common.exceptions import ModelNotTrainedException, InvalidArgumentException
from models.support_vector import SVM


class TestSVM:
    def setup_method(self):
        self.svm = SVM()

    def test_learn(self):
        features = np.array([[1, 2], [2, 3], [3, 4]])
        targets = np.array([1, -1, 1])
        self.svm.learn(features, targets, epochs=100)
        assert self.svm.weights is not None

    def test_infer(self):
        features = np.array([[1, 2], [2, 3], [3, 4]])
        targets = np.array([1, -1, 1])
        self.svm.learn(features, targets, epochs=1)
        # can't have an exact test for an approximation algorithm
        result = self.svm.infer(features[0])
        assert result.shape == targets[0].shape
        results = self.svm.infer_inputs(features)
        assert results.shape == targets.shape

    def test_learn_invalid_inputs(self):
        features = np.array([[1, 2], [2, 3], [3, 4]])
        targets = np.array([1, -1, 1])
        with pytest.raises(InvalidArgumentException):
            self.svm.learn(features, targets, epochs=0)
        with pytest.raises(InvalidArgumentException):
            self.svm.learn(features, targets, epochs=-1)
        with pytest.raises(InvalidArgumentException):
            self.svm.learn(features, targets[:-1], epochs=100)

    def test_infer_untrained_model(self):
        with pytest.raises(ModelNotTrainedException):
            self.svm.infer(np.array([1, 2]))
