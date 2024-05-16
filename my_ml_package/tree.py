import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier_sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.tree import plot_tree

def plot_tree(Xtrain, ytrain, feature_cols, max_depth, show_uncertainty=False, criterion="entropy",  min_samples_leaf=1):

    treeclf = DecisionTreeClassifier_sklearn(max_depth=max_depth, random_state=1, min_samples_leaf=min_samples_leaf, criterion=criterion)
    treeclf.fit(Xtrain, ytrain)
    plt.figure(figsize=(20,18))
    tree.plot_tree(treeclf, feature_names=feature_cols,  impurity=show_uncertainty, class_names=['Died', 'Survived'], filled=True)
    plt.show()


import numpy as np
from collections import deque

# STACK - DATA STRUCTURE
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y, feature_idx2name =None):
        self.tree = self._build_tree_breadth_first(X, y, feature_idx2name)

    def _build_tree_depth_first(self, X, y, feature_idx2name=None, depth=0):
        assert type(X) == np.ndarray
        assert type(y) == np.ndarray

        num_samples = X.shape[0]
        if num_samples <= 1 or len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # Compute the information gain for each feature and choose the best one
        best_feature_idx, best_threshold = self._best_criteria(X, y, X.shape[1])
        if feature_idx2name:
            print(f'Depth: {depth}, Best feature: {feature_idx2name[best_feature_idx]}, Best threshold: {best_threshold}')
            print('\n')
        
        # recursively build the subtrees
        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_threshold)
        left_subtree = self._build_tree_depth_first(X[left_idxs, :], y[left_idxs],  feature_idx2name, depth+1)
        right_subtree = self._build_tree_depth_first(X[right_idxs, :], y[right_idxs], feature_idx2name, depth+1)
        return self.Node(best_feature_idx, best_threshold, left_subtree, right_subtree)
    
    def _build_tree_breadth_first(self, X, y, feature_idx2name=None):
        queue = deque()
        root = self.Node()
        queue.append((root, X, y, 0))  # Node, features, target, depth

        while queue:
            current_node, data, labels, depth = queue.popleft()

            if len(np.unique(labels)) == 1 or depth == self.max_depth:
                current_node.value = self._most_common_label(labels)
                continue

            best_feature_idx, best_threshold = self._best_criteria(data, labels, X.shape[1])
            if feature_idx2name:
                print(f'Depth: {depth}, Best feature: {feature_idx2name[best_feature_idx]}, Best threshold: {best_threshold}')
                print('\n')
            if best_feature_idx is None:
                current_node.value = self._most_common_label(labels)
                continue

            left_idxs, right_idxs = self._split(data[:, best_feature_idx], best_threshold)
            current_node.feature = best_feature_idx
            current_node.threshold = best_threshold
            current_node.left = self.Node()
            current_node.right = self.Node()

            queue.append((current_node.left, data[left_idxs, :], labels[left_idxs], depth + 1))
            queue.append((current_node.right, data[right_idxs, :], labels[right_idxs], depth + 1))

        return root

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def _best_criteria(self, X, y, num_features):
        best_gain = -1
        select_feature, select_threshold = None, None
        for feature in range(num_features):
            gain, threshold = feature_importance(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                select_feature, select_threshold = feature, threshold
        
        return select_feature, select_threshold

    def _split(self, feature, threshold):
        left_idxs = np.where(feature <= threshold)[0]
        right_idxs = np.where(feature > threshold)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

def information_gain(y_parent, left_child, right_child, method='entropy'):
    """Calculate the information gain from a parent to its children.
    Equation: IG(Y, X) = H(Y) - Σ |Yv| / |Y| * H(Yv), where X is the feature, Y is the parent node, and Yv is the child node of X = V.
    args:
        y_parent: a list of classes representing the parent node
        left_child: a list of classes representing the left child node
        right_child: a list of classes representing the right child node
    """
    uncertainty_measure = entropy if method == 'entropy' else gini_impurity
    # calculate the entropy of the parent
    parent_uncertainty = uncertainty_measure(y_parent)
    # calculate the entropy of the children
    left_uncertainty = uncertainty_measure(left_child)
    right_uncertainty = uncertainty_measure(right_child)
    # calculate the information gain
    p = float(len(left_child)) / (len(left_child) + len(right_child))
    gain = parent_uncertainty - p * left_uncertainty -  (1 - p)* right_uncertainty
    return gain

def entropy(y):
    """Calculate the entropy for a list of classes.
    Equation: H(Y) = -Σ Pr(Y=yc) * log2(Pr(Y=yc))
    args:
        y: a list of values for ground truths (classes)
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def feature_importance(X, y, feature):
    possible_thresholds = np.unique(X[:, feature])
    best_gain = 0
    best_threshold = None
    for threshold in possible_thresholds:
        left = y[X[:, feature] <= threshold]
        right = y[X[:, feature] > threshold]
        if len(left) == 0 or len(right) == 0:
            continue
        gain = information_gain(y, left, right)
        if gain > best_gain:

            best_gain = gain
            best_threshold = threshold
    return best_gain, best_threshold

