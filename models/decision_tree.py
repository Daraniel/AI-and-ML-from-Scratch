from __future__ import annotations

from typing import Union

import numpy as np

from common.base_model import BaseModel, Classifier
from common.exceptions import (InvalidArgumentException,
                               ModelNotTrainedException)
from common.utils import ImpurityFunctions


class DecisionNode:
    def __init__(self, label, attribute: Union[np.ndarray[int], int] = -1, children=None):
        """
        Creates a decision tree node
        :param label: label of this node
        :param attribute: attribute assigned to this node
        :param children: children of this node
        """
        self.label = label
        self.attribute = attribute
        self.children = children


class DecisionTree(Classifier, BaseModel):
    def __init__(
            self, impurity_function=ImpurityFunctions.entropy, verbose: bool = False
    ):
        if not callable(impurity_function):
            raise InvalidArgumentException(
                f"split_criterion must be a callable not a {type(impurity_function)}"
            )
        self.impurity_function = impurity_function
        self.verbose = verbose
        self.root = None

    def learn(self, features: np.array, targets: np.array, max_depth: int = None):
        """
        train the model using features to predict targets, creating the branches will continue until a pure branch has
        been reached, no attributes are left or max_depth is reached
        :param features: features to train the model on
        :param targets: the most common classes of each feature
        :param max_depth: max depth of the tree, optional, if passed the model will be trained until max depth has
        been reached and will stop it
        """
        if max_depth is not None:
            if not isinstance(max_depth, int):
                raise InvalidArgumentException("max_depth must be an integer")
            if max_depth <= 0:
                raise InvalidArgumentException("max_depth must be a positive number")
        self.root = self._learn(features, targets, max_depth)

    def _learn(
            self,
            features: np.array,
            targets: np.array,
            max_depth: int = None,
            attributes=None,
    ) -> DecisionNode:
        """
        train the model using features to predict targets, creating the branches will continue until a pure branch has
        been reached, no attributes are left or max_depth is reached
        :param features: features to train the model on
        :param targets: the most common classes of each feature
        :param max_depth: max depth of the tree, optional, if passed the model will be trained until max depth has
        been reached and will stop it
        :param attributes: attributes of the current branch, optional
        """
        if attributes is None:
            attributes = {
                i: np.unique(features[:, i]) for i in range(features.shape[1])
            }

        label, is_pure = self._get_the_most_common_target_class(targets)
        if is_pure:
            if self.verbose:
                print(
                    f"Leaf node with label {label} was created as it has no siblings (is pure)"
                )
            return DecisionNode(label)

        if len(attributes) == 0:
            if self.verbose:
                print(
                    f"Leaf node with label {label} was created as attributes have been exhausted"
                )
            return DecisionNode(label)

        if max_depth is not None and max_depth == 1:
            if self.verbose:
                print(
                    f"Leaf node with label {label} was created as max depth has been reached"
                )
            return DecisionNode(label)

        (
            index_of_maximum_impurity_reduction,
            maximum_impurity_reduction_attribute,
        ) = self._get_split_attribute(features, targets, attributes)

        values = attributes.pop(index_of_maximum_impurity_reduction)
        splits = [
            features[:, index_of_maximum_impurity_reduction] == value
            for value in values
        ]
        branches = {}

        for value, split in zip(values, splits):
            if not np.any(split):
                if self.verbose:
                    print(
                        f"Empty split for value {value} of attribute {maximum_impurity_reduction_attribute}"
                    )
                branches[value] = DecisionNode(label)
            else:
                if self.verbose:
                    print(
                        f"Recursion for value {value} of attribute {maximum_impurity_reduction_attribute}"
                    )
                    if max_depth is not None:
                        branches[value] = self._learn(
                            features[split, :],
                            targets[split],
                            max_depth=max_depth - 1,
                            attributes=attributes,
                        )
                    else:
                        branches[value] = self._learn(
                            features[split, :],
                            targets[split],
                            max_depth=max_depth,
                            attributes=attributes,
                        )
        attributes[index_of_maximum_impurity_reduction] = values
        return DecisionNode(
            label, attribute=index_of_maximum_impurity_reduction, children=branches
        )

    def infer(self, features: np.array) -> np.array:
        """
        find the most common class for each feature in the features
        :param features: features to find their classes
        :return: list of the most common classes for each feature
        """
        if self.root is None:
            raise ModelNotTrainedException("Please train the model before using it!")

        return [self._infer(feature, self.root) for feature in features]

    @staticmethod
    def _infer(feature, node: DecisionNode):
        """
        find the most common class for a single feature in a recursive manner
        :param feature: feature to find its classes
        :param node: a node in the decision tree, it will be root in the first iteration and after that it will follow
        the branch that this feature belongs to in the tree
        :return: most common class of this feature
        """
        if not node.children:
            return node.label
        else:
            child_node = node.children[feature[node.attribute]]
            return DecisionTree._infer(feature, child_node)

    @staticmethod
    def _get_the_most_common_target_class(targets):
        """
        finds the most common target class
        :param targets: concept vector, number_of_records * 1
        :return: the most common class and a flag indicating whether this target is pure or not (whether tho most common
        class is singular or not)
        """
        unique_target, number_of_unique_target_occurrences = np.unique(
            targets, return_counts=True
        )
        the_most_common_target_class = unique_target[
            np.argmax(number_of_unique_target_occurrences)
        ]

        splittings_is_pure = len(unique_target) == 1
        return the_most_common_target_class, splittings_is_pure

    def _get_split_attribute(self, features, targets, attributes):
        """
        calculates the best attribute for the given features with maximum the impurity reduction
        :param features: input, number_of_records * number_of_records
        :param targets: concept vector, number_of_records * 1
        :param attributes: mapping of attributes indexes to their names
        """

        impurity_reduction = [0] * features.shape[0]
        for attribute_index in attributes.keys():
            impurity_reduction[attribute_index] = self._get_impurity_reduction(
                features, attribute_index, targets
            )

        index_of_maximum_impurity_reduction = np.argmax(impurity_reduction)

        if self.verbose:
            print(
                f"Impurity reduction ordered by attributes is as follows: {impurity_reduction}"
            )
            print(
                f"Maximum impurity reduction of {impurity_reduction[index_of_maximum_impurity_reduction]} "
                f"for the attribute {index_of_maximum_impurity_reduction}"
            )

        return index_of_maximum_impurity_reduction, {
            impurity_reduction[index_of_maximum_impurity_reduction]
        }

    def _get_impurity_reduction(
            self, features: np.ndarray, attribute_to_reduce: np.ndarray, targets: np.ndarray
    ):
        """
        calculates the impurity reduction for the given input
        :param features: input, number_of_records * number_of_records
        :param attribute_to_reduce: column of indexes to reduce their impurity
        :param targets: concept vector, number_of_records * 1
        :return: impurity reduction
        """
        relative_frequency = DecisionTree._get_relative_frequency(
            occurrences=features.shape[0], targets=targets
        )

        impurity = self.impurity_function(relative_frequency)

        if self.verbose:
            print(f"Impurity {impurity} calculated for {relative_frequency}")

        impurity_of_splits = self._evaluate_impurity_of_splittings(
            features, attribute_to_reduce, targets
        )

        impurity_reduction = impurity - np.sum(impurity_of_splits)
        if self.verbose:
            print(f"Estimated impurity reduction of {impurity_reduction}")
        return impurity_reduction

    def _evaluate_impurity_of_splittings(self, features, attribute_to_reduce, targets):
        """
        calculate the impurities for different splits of features induced by attributes_to_reduce
        :param features: input, number_of_records * number_of_records
        :param attribute_to_reduce: column of indexes to reduce their impurity
        :param targets: concept vector, number_of_records * 1
        :return: list of impurities
        """
        splittings = []
        for unique_feature in np.unique(features[:, attribute_to_reduce]):
            wanted_features_mask = features[:, attribute_to_reduce] == unique_feature
            number_of_wanted_features = np.sum(wanted_features_mask)
            relative_frequency = DecisionTree._get_relative_frequency(
                occurrences=number_of_wanted_features,
                targets=targets,
                mask=wanted_features_mask,
            )
            impurity = (
                               number_of_wanted_features / features.shape[0]
                       ) * self.impurity_function(relative_frequency)
            splittings.append(impurity)
            if self.verbose:
                print(
                    f"Impurity {impurity} calculated for the attributes {attribute_to_reduce} "
                    f"with value {unique_feature} and relative frequency of {relative_frequency}"
                )

        return splittings

    @staticmethod
    def _get_relative_frequency(occurrences, targets, mask=None):
        """
        calculates the relative frequency of each input in the target (please note that instead of directly using the
        input, it is represented here by its occurrences)
        :param occurrences: size of the input
        :param targets: target to calculate the relative frequency based on it
        :param mask: optional, mask to apply to the target
        :return: relative frequency of the input based on the target
        """
        unique_targets = np.unique(targets)
        if mask is not None:
            targets = targets[mask]
        return (1 / occurrences) * np.ndarray(
            [np.sum(targets == unique_element) for unique_element in unique_targets]
        )
