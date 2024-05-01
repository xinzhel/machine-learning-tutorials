import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters:
    - z: numpy array of shape (n_samples,), the input to the sigmoid function.

    Returns:
    - sigmoid: numpy array of shape (n_samples,), the output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))

def logistic_regression_for_binary(X, y,  W=None, b=None, lr=0.01, num_iter=1000, reg_lambda=0, reg_type=None):
    """
    Perform logistic regression on the dataset (X, y) using gradient descent.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), the input data.
    - y: numpy array of shape (n_samples,), the target values.
    - lr: float, the learning rate for gradient descent.
    - num_iter: int, the number of iterations for gradient descent.

    Returns:
    - w: numpy array of shape (n_features,), the weights of the logistic regression model.
    - b: float, the bias of the logistic regression model.
    """
    n_samples, n_features = X.shape
    if W is None:
        assert b is None, "If W is not provided, b should not be provided as well."
        beta1 = np.zeros(n_features)
        beta0 = 0
    else:
        beta1 = W
        beta0 = b

    for _ in range(num_iter):
        # Compute the predicted values
        y_pred = sigmoid(np.dot(X, beta1) + beta0)

        # Compute the gradients
        if reg_type == "l2":
            d_beta1 = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + reg_lambda * beta1)
        elif reg_type == "l1":
            d_beta1 = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + reg_lambda * np.sign(beta1))
        else:
            d_beta1 = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        
        d_beta0 = (1 / n_samples) * np.sum(y_pred - y)

        # Update the weights and bias
        beta1 -= lr * d_beta1
        beta0 -= lr * d_beta0

    return beta1, beta0

def logistic_regression_for_multiclass(X, y, W=None, b=None, lr=0.01, num_iter=1000, reg_lambda=0, reg_type=None):
    """
    Perform logistic regression for multiclass classification using one-vs-all strategy.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), the input data.
    - y: numpy array of shape (n_samples,), the target values.
    - lr: float, the learning rate for gradient descent.
    - num_iter: int, the number of iterations for gradient descent.

    Returns:
    - W: numpy array of shape (n_features, n_classes), the weights of the logistic regression model.
    - b: numpy array of shape (n_classes,), the biases of the logistic regression model.
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    if W is None:
        assert b is None, "If W is not provided, b should not be provided as well."
        W = np.zeros((n_features, n_classes))
        b = np.zeros(n_classes)

    for i in range(n_classes):
        assert i in y, f"Class {i} is not present in the target values."
        # Convert the multiclass problem into a binary classification problem
        binary_y = np.where(y == i, 1, 0)

        # Perform binary logistic regression
        W[:, i], b[i] = logistic_regression_for_binary(X, binary_y,  W[:, i], b[i], lr, num_iter, reg_lambda, reg_type)

    return W, b

def lr_predict_binary(X, beta1, beta0):
    """
    Predict the target values using the logistic regression model.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), the input data.
    - w: numpy array of shape (n_features,), the weights of the logistic regression model.
    - b: float, the bias of the logistic regression model.

    Returns:
    - y_pred: numpy array of shape (n_samples,), the predicted target values.
    """
    return sigmoid(np.dot(X, beta1) + beta0)

def lr_predict_multiclass(X, W, b):
    """
    Predict the target values using the logistic regression model.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), the input data.
    - W: numpy array of shape (n_features, n_classes), the weights of the logistic regression model.
    - b: numpy array of shape (n_classes,), the biases of the logistic regression model.

    Returns:
    - y_pred: numpy array of shape (n_samples,), the predicted target values.
    """
    n_samples = X.shape[0]
    n_classes = W.shape[1]
    y_pred = np.zeros((n_samples, n_classes))

    for i in range(n_classes):
        y_pred[:, i] = lr_predict_binary(X, W[:, i], b[i])

    return np.argmax(y_pred, axis=1)

