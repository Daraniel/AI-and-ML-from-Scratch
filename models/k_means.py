import random

import numpy as np

from common.base_model import BaseModel, Classifier
from common.utils import DistanceMetrics


class KMeansClassifier(BaseModel, Classifier):
    def __init__(self, distance_metric=DistanceMetrics.euclidean_distance):
        """
        :param distance_metric: distance metric used by the k-means clustering
        """
        self.central_points = None
        self.clusters = {}
        self.distance_metric = distance_metric

    def _get_start_central_points(self, points: np.ndarray, k: int):
        """
        get the initial central points for the k-means clustering algorithm
        :param points: points to find their center
        :param k: number of centers (clusters)
        """
        self.central_points = []
        while len(self.central_points) < k:
            self.central_points = points[random.sample(range(0, len(points)), k - len(self.central_points))]
            self. central_points = np.unique(self.central_points, axis=0)

    def _get_new_central_points(self):
        """
        calculates the new central points by calculating mean of the points in each cluster
        """
        self.central_points = np.array([np.mean(self.clusters[key], axis=0) for key in sorted(self.clusters.keys())])

    def _get_clusters(self, points: np.ndarray):
        """
        finds the cluster that each point belongs to. this cluster is the cluster that its center has the least
        distance_metric units distance from that point
        :param points: points to find their cluster
        """
        self.clusters = {}
        closest_points = np.argmin(self.distance_metric(points, self.central_points), axis=1)

        for i, closest_point in enumerate(closest_points):
            if closest_point not in self.clusters:
                self.clusters[closest_point] = [points[i]]
            else:
                self.clusters[closest_point].append(points[i])

    def _search_can_continue(self, previous: np.ndarray, current: np.ndarray,
                             max_single_element_change_threshold: float = 0, max_average_change_threshold: float = 0):
        """
        determined whether search should continue or not. it can continue when the average of central points has moved
        more than max_average_change_threshold or a single point has moved more than max_single_element_change_threshold
        :param previous: previous central points
        :param current: current central points
        :param max_single_element_change_threshold: max ignorable movement for each single point
        :param max_average_change_threshold: max average movement for all points
        :return: whether the search can continue or not
        """
        change = self.distance_metric(previous, current).diagonal()
        return (np.max(change) > max_single_element_change_threshold
                and np.average(np.abs(change)) > max_average_change_threshold)

    def learn(self, points: np.ndarray, k: int, max_iterations: int = 10,
              max_single_element_change_threshold: float = 0, max_average_change_threshold: float = 0):
        """
        trains the k-means clustering algorithm and finds the central points of each cluster
        :param points: points to classify
        :param k: number of clusters
        :param max_iterations: maximum iterations for the algorithm
        :param max_single_element_change_threshold: max ignorable movement for each single point
        :param max_average_change_threshold: max average movement for all points
        :return: central points of the clusters
        """
        previous_central_points = np.zeros((k, points.shape[1]))
        self._get_start_central_points(points, k)
        self._get_clusters(points)

        while self._search_can_continue(previous_central_points, self.central_points):
            self._get_clusters(points)
            previous_central_points = self.central_points.copy()
            self._get_new_central_points()

        return self.central_points

    def infer(self, points: np.array) -> np.array:
        """
        classifies the points and finds which cluster they belong to
        :param points: points to classify
        :return: classes of the points
        """
        self._get_clusters(points)
        return self.clusters
