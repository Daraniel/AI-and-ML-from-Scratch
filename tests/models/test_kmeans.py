import numpy as np
import pytest

from models.k_means import KMeansClassifier


class TestKMeansClassifier:
    @pytest.fixture
    def kmeans_classifier(self):
        return KMeansClassifier()

    def test_learn(self, kmeans_classifier):
        points = np.array([[1, 2], [1, 4], [2, 2], [2, 4], [3, 2], [3, 4]])
        kmeans_classifier.learn(points, 2)
        assert len(kmeans_classifier.central_points) == 2
        assert len(kmeans_classifier.clusters) == 2

    def test_infer(self, kmeans_classifier):
        points = np.array([[1, 2], [1, 4], [2, 2], [2, 4], [3, 2], [3, 4]])
        kmeans_classifier.learn(points, 2)
        classes = kmeans_classifier.infer(points)
        # can't have an exact test for an approximation algorithm
        assert set(classes) == {0, 1}
        assert len(classes[0]) > 0
        assert len(classes[1]) > 0
