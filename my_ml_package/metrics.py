from sklearn.model_selection import KFold, cross_val_score
import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Parameters:
    - y_true: numpy array of true target values.
    - y_pred: numpy array of predicted target values.

    Returns:
    - mse: Mean Squared Error of the model's predictions.
    """
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """
    Parameters:
    - y_true: numpy array of true target values.
    - y_pred: numpy array of predicted target values.

    Returns:
    - mae: Mean Absolute Error of the model's predictions.
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    Parameters:
    - y_true: numpy array of true target values.
    - y_pred: numpy array of predicted target values.

    Returns:
    - r2: R-squared value of the model's predictions.
    """
    # Calculate the mean of the true target values
    y_true_mean = np.mean(y_true)

    # Calculate the total sum of squares
    ss_total = np.sum((y_true - y_true_mean) ** 2)

    # Calculate the residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate the R-squared value
    r2 = 1 - (ss_res / ss_total)

    return r2

def variance_in_cv_scores(model, X, y):
    # Define a cross-validation strategy (e.g., 5-fold cross-validation)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and calculate scores for each fold
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

    # Calculate and return variability measures
    std_deviation = np.std(scores)
    return std_deviation
