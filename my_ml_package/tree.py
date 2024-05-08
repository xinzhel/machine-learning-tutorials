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