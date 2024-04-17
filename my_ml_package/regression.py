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

def ols_estimate_for_linear_regression(X, Y):
    """
    Calculates the Ordinary Least Squares (OLS) estimates of the intercept (beta_0) 
    and slope (beta_1) for a simple linear regression model. The model assumes a 
    linear relationship between a single independent variable X and a dependent 
    variable Y, expressed as Y = beta_0 + beta_1*X + epsilon, where epsilon represents 
    the error term of the model.

    The OLS method aims to minimize the sum of the squared differences between the 
    observed values in the dataset and those predicted by the linear model. This 
    minimization leads to the derivation of the closed-form solutions for beta_0 
    (intercept) and beta_1 (slope) as follows:

    beta_1 = sum((X_i - mean(X)) * (Y_i - mean(Y))) / sum((X_i - mean(X))^2)
    beta_0 = mean(Y) - beta_1 * mean(X)

    This equals to `np.polyfit(X, Y, 1)` and
    ```
    from sklearn.linear_model import LinearRegression
    my_model = LinearRegression()
    my_model.fit(x.reshape(-1, 1), y)
    print ("Slope: ", my_model.coef_[0])
    print ("Intercept: ", my_model.intercept_)
    ```

    Parameters:
    - X: numpy array of independent variable values.
    - Y: numpy array of dependent variable values.

    Returns:
    - beta_0: Estimated intercept of the regression line.
    - beta_1: Estimated slope of the regression line.
    """
      # numerator = np.sum(np.multiply(X - X_mean, Y - Y_mean))
    # denominator = np.sum(np.power(X - X_mean, 2))
    # beta_1 = numerator / denominator

    # Mean of X and Y
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    # Calculate beta_1 (slope)
    beta_1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean) ** 2)

    # Calculate beta_0 (intercept)
    beta_0 = Y_mean - beta_1 * X_mean

    return beta_0, beta_1

if __file__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/house-prices/train.csv") # from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
    x = df['GrLivArea'] # Above grade (ground) living area square feet
    y = df['SalePrice']
    ols_estimate_for_linear_regression(x, y)
