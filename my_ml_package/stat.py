
import numpy as np
def var(x):
    """ Sample Variance of a variable.
    It measures how spread out the data points are around the mean by
     the average of the squared differences of each data point from the sample mean.
    Equation: var(x) = Σ (x_i - mean(x))^2 / n
    Equivalent to: var(x) = E[(x - E[x])^2]
    Simplified to: var(x) = E[x^2] - E[x]^2
    Args:
        x (np.ndarray): A NumPy array."""
    
    return np.sum((x- x.mean())**2)/len(x)

def covar(x, y):
    """ Covariance of two variables.
    It measures how two variables change together by 
     the average of the product of the differences of each data point from the sample mean.
    Equation: cov(x, y) = Σ (x_i - mean(x)) * (y_i - mean(y)) / n
    Equivalent to: cov(x, y) = E[(x - E[x]) * (y - E[y])] 
    Simplified to: cov(x, y) = E[xy] - E[x]E[y]
    Args:
        x, y (np.ndarray): Two NumPy arrays of the same length."""
    return np.sum((x - x.mean())*(y - y.mean()))/len(x)

def covar_matrix(X):
    """ Covariance matrix of a dataset.
    It is a square matrix that describes the covariance between two or more variables in a dataset.
    Args:
        X (np.ndarray): A NumPy array of shape (n_samples, n_features)."""
    n_samples, n_features = X.shape
    covar_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            covar_matrix[i, j] = covar(X[:, i], X[:, j])
    return covar_matrix

def covar_matrix_efficient(X):
    """ Covariance matrix of a dataset.
    Equation: cov_matrix = Σ (X_i - mean(X, axis=0)).T * (X_i - mean(X, axis=0)) / n
    Simplified to: cov_matrix = E[X.T * X] - E[X].T * E[X]
    """
    mu = X.mean(axis=0) # shape: (n_features,)
    sigma = X.std(axis=0)  # shape: (n_features,) 
    Xnorm = (X - mu)/sigma # data shape: (n_sample, n_features)
    return np.dot(Xnorm.T, Xnorm)/len(Xnorm) 

def find_position_for_percentile(percentile, sorted_data):
    """ Returns the position of a given percentile in a sorted list of data.
    Percentile: A value below which a given percentage of data falls.
    """
    n = len(sorted_data)
    position = (percentile/100) * (n + 1)
    return position


def find_percentile(percentile, sorted_data):
    """ Returns the value of a given percentile in a sorted list of data.
    Percentile: A value below which a given percentage of data falls.
    """
    position = find_position_for_percentile(percentile, sorted_data)
    if position.is_integer():
        return sorted_data[int(position) - 1]
    else:
        k = int(position) 
        fraction = position - k
        return sorted_data[k - 1] + fraction * (sorted_data[k] - sorted_data[k - 1]) # Linear interpolation

# Interquartile Range (IQR)
def find_median(sorted_data):
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        return sorted_data[n//2]

def calculate_quartiles(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Calculate Q1
    lower_half = sorted_data[:n//2]
    Q1 = find_median(lower_half)
    
    # calculate Q2
    Q2 = find_median(sorted_data)

    # Calculate Q3
    if n % 2 == 0:
        upper_half = sorted_data[n//2:]
    else: # len(data)%2 == 1
        upper_half = sorted_data[n//2+1:]
    Q3 = find_median(upper_half)
    
    return Q1, Q2, Q3

def calculate_iqr_bounds(data):
    Q1, _, Q3 = calculate_quartiles(data)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound
