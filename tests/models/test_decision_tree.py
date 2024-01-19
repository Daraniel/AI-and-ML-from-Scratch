import numpy as np

from models.decision_tree import DecisionTree


class TestDecisionTree:
    def test_learn(self):
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        targets = np.array([0, 1, 0])
        max_depth = 2
        decision_tree = DecisionTree(verbose=True)
        assert decision_tree.root is None
        decision_tree.learn(features, targets, max_depth)
        assert decision_tree.root is not None
        assert decision_tree.root.label == 0
        assert decision_tree.root.attribute == 2
        assert len(decision_tree.root.children) == 3

    def test_infer(self):
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        targets = np.array([0, 1, 0])
        max_depth = 2
        decision_tree = DecisionTree(verbose=True)
        decision_tree.learn(features, targets, max_depth)
        inferred_classes = decision_tree.infer(features)
        assert inferred_classes == [0, 1, 0]

    def test_get_the_most_common_target_class(self):
        targets = np.array([0, 1, 0, 1, 2, 1])
        expected_result = 1
        expected_is_pure = False
        actual_result, actual_is_pure = DecisionTree._get_the_most_common_target_class(
            targets
        )
        assert actual_result == expected_result
        assert actual_is_pure == expected_is_pure
