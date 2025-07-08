import pandas as pd
from my_ml_package.models.regression import ols_estimate_for_linear_regression
from my_ml_package.metrics import mean_squared_error
df = pd.read_csv("data/house-prices/train.csv") # from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
x = df['GrLivArea'] # Above grade (ground) living area square feet
y = df['SalePrice']
beta_0, beta_1 = ols_estimate_for_linear_regression(x, y)
print(f"Intercept: {beta_0}, Slope: {beta_1}")
predictions = beta_0 + beta_1 * x
error  = mean_squared_error(y, predictions)
print(f"Mean Squared Error: {error}")


