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
    w = np.zeros(n_features)
    b = 0

    for _ in range(num_iter):
        # Compute the predicted values
        y_pred = sigmoid(np.dot(X, w) + b)

        # Compute the gradients
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Update the weights and bias
        w -= lr * dw
        b -= lr * db

    return w, b