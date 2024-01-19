import numpy as np

from common.utils import AverageNormalizer2D, ActivationFunctions, ActivationFunctionsForNN, EvaluationMetrics, \
    ImpurityFunctions


class TestAverageNormalizer2D:
    def test_calculate(self):
        normalizer = AverageNormalizer2D()
        x = np.array([[1, 2], [3, 4], [5, 6]])
        normalizer.calculate(x)
        assert normalizer.average is not None
        assert normalizer.std is not None

    def test_normalize(self):
        normalizer = AverageNormalizer2D()
        x = np.array([[1, 2], [3, 4], [5, 6]])
        normalizer.calculate(x)
        normalized = normalizer.normalize(x)
        assert normalized.shape == x.shape

    def test_denormalize(self):
        normalizer = AverageNormalizer2D()
        x = np.array([[1, 2], [3, 4], [5, 6]])
        normalizer.calculate(x)
        normalized = normalizer.normalize(x)
        denormalized = normalizer.denormalize(normalized)
        assert denormalized.shape == x.shape


class TestActivationFunctions:
    def test_sigmoid(self):
        x = np.array([1, 2, 3])
        sigmoid = ActivationFunctions.sigmoid(x)
        assert sigmoid.shape == x.shape
        assert np.all(sigmoid >= 0) and np.all(sigmoid <= 1)

    def test_relu(self):
        x = np.array([-1, 0, 1])
        relu = ActivationFunctions.relu(x)
        assert relu.shape == x.shape
        assert np.all(relu >= 0)


class TestActivationFunctionsForNN:
    def test_relu(self):
        relu = ActivationFunctionsForNN.Relu()
        x = np.array([-1, 0, 1])
        forward = relu.forward(x)
        assert forward.shape == x.shape
        assert np.all(forward >= 0)

        backward = relu.backward(x)
        assert backward.shape == x.shape
        np.testing.assert_allclose(backward, np.array([0., 0., 1.]))

    def test_sigmoid(self):
        sigmoid = ActivationFunctionsForNN.Sigmoid((3,))
        x = np.array([1, 2, 3])
        forward = sigmoid.forward(x)
        assert forward.shape == x.shape
        assert np.all(forward >= 0) and np.all(forward <= 1)

        backward = sigmoid.backward(x)
        assert backward.shape == x.shape
        np.testing.assert_allclose(backward, np.array([0.19661193, 0.20998717, 0.13552998]))

    def test_softmax(self):
        expected_results = [
            np.array([[4.48309e-06, 2.71913e-06, 2.01438e-06, 3.31258e-05],
                      [4.48309e-06, 6.06720e-07, 1.80861e-03, 3.31258e-05],
                      [1.21863e-05, 2.68421e-01, 7.29644e-01, 3.31258e-05]]),
            np.array([[2.11942e-01, 1.01300e-05, 2.75394e-06, 3.33333e-01],
                      [2.11942e-01, 2.26030e-06, 2.47262e-03, 3.33333e-01],
                      [5.76117e-01, 9.99988e-01, 9.97525e-01, 3.33333e-01]]),
            np.array([[1.05877e-01, 6.42177e-02, 4.75736e-02, 7.82332e-01],
                      [2.42746e-03, 3.28521e-04, 9.79307e-01, 1.79366e-02],
                      [1.22094e-05, 2.68929e-01, 7.31025e-01, 3.31885e-05]])
        ]
        for axis, expected_result, sum_result in zip([None, 0, 1], expected_results, [1, 4, 3]):
            relu = ActivationFunctionsForNN.Softmax(axis=axis)
            x = np.array([[1, 0.5, 0.2, 3],
                          [1, -1, 7, 3],
                          [2, 12, 13, 3]])

            forward = relu.forward(x)
            assert forward.shape == x.shape
            assert np.all(forward >= 0)
            np.testing.assert_allclose(forward.sum(), sum_result)
            np.testing.assert_allclose(forward, expected_result, rtol=0, atol=1e-6)

            backward = relu.backward(x)
            np.testing.assert_equal(backward, x)


class TestEvaluationMetrics:
    def test_rmse(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 3, 4])
        rmse = EvaluationMetrics.rmse(x, y)
        assert rmse >= 0

    def test_hinge_loss(self):
        x = np.array([1, 2, 3])
        y = np.array([2, 3, 4])
        hinge_loss = EvaluationMetrics.hinge_lost(x, y)
        assert np.all(hinge_loss >= 0)

    def test_cross_entropy_loss(self):
        yhat = np.array([0.1, 0.2, 0.3])
        y = np.array([0, 1, 0])
        cross_entropy_loss = EvaluationMetrics.cross_entropy_loss(yhat, y)
        assert cross_entropy_loss >= 0

    def test_euclidean_distance(self):
        first = np.array([[1, 2], [3, 4]])
        second = np.array([[5, 6], [7, 8]])
        distance = EvaluationMetrics.euclidean_distance(first, second)
        assert distance.shape == (2, 2)
        np.testing.assert_allclose(distance, np.array([[5.65685425, 8.48528137], [2.82842712, 5.65685425]]))


class TestImpurityFunctions:
    def test_gini(self):
        x = np.array([0.2, 0.4, 0.4, 0.2])
        expected_result = 0.5999999999999999
        assert ImpurityFunctions.gini(x) == expected_result

    def test_entropy(self):
        x = np.array([0.2, 0.4, 0.4, 0.2])
        expected_result = 1.9863137138648346
        assert ImpurityFunctions.entropy(x) == expected_result

    def test_mis_classification(self):
        x = np.array([0.2, 0.4, 0.4, 0.2])
        expected_result = 0.6
        assert ImpurityFunctions.mis_classification(x) == expected_result
