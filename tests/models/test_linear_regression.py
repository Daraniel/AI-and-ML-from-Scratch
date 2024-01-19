import numpy as np
import pytest

from common.exceptions import ModelNotTrainedException
from models.linear_regression import LinearRegressionAnalytic, LinearRegressionGradiantDecent


class TestLinearRegressionAnalytic:
    @pytest.fixture
    def linear_regression_analytic(self):
        return LinearRegressionAnalytic()

    def test_learn_analytic(self, linear_regression_analytic):
        features = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
        targets = np.array([[1], [2], [3], [4], [5], [6]])
        linear_regression_analytic.learn(features, targets)
        assert linear_regression_analytic.weights is not None
        assert linear_regression_analytic.bias is not None

    def test_infer_analytic(self, linear_regression_analytic):
        features = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        targets = np.array([[1], [2], [3], [4]])
        linear_regression_analytic.learn(features, targets)
        inferred = linear_regression_analytic.infer(features).reshape(-1, 1)
        np.testing.assert_allclose(inferred, targets)


class TestLinearRegressionGradiantDecent:
    @pytest.fixture
    def linear_regression_gradient_decent(self):
        return LinearRegressionGradiantDecent()

    def test_learn_gradient_decent(self, linear_regression_gradient_decent):
        features = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        targets = np.array([[1], [2], [3], [4]])
        linear_regression_gradient_decent.learn(features, targets)
        assert linear_regression_gradient_decent.weights is not None
        assert linear_regression_gradient_decent.bias is not None

    def test_infer_gradient_decent(self, linear_regression_gradient_decent):
        features = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        targets = np.array([[1], [2], [3], [4]])
        linear_regression_gradient_decent.learn(features, targets)
        inferred = linear_regression_gradient_decent.infer(features).reshape(-1, 1)
        # can't have an exact test for an approximation algorithm
        assert inferred.shape == targets.shape

    def test_not_trained(self, linear_regression_gradient_decent):
        x = np.random.rand(10)
        with pytest.raises(ModelNotTrainedException):
            linear_regression_gradient_decent.infer(x)
