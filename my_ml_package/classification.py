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

def logistic_regression(X, y, lr=0.01, num_iter=1000):
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
    beta1 = np.zeros(n_features)
    beta0 = 0

    for _ in range(num_iter):
        # Compute the predicted values
        y_pred = sigmoid(np.dot(X, beta1) + beta0)

        # Compute the gradients
        d_beta1 = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        d_beta0 = (1 / n_samples) * np.sum(y_pred - y)

        # Update the weights and bias
        beta1 -= lr * d_beta1
        beta0 -= lr * d_beta0

    return beta1, beta0

def lr_predict(X, beta1, beta0):
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