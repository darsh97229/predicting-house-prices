import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the FRED house price index dataset
df = pd.read_csv('../Data/CSUSHPINSA.csv')

# Split the data into features (X) and target variable (y)
X = df.drop(['DATE', 'CSUSHPINSA'], axis=1) # Drop the 'DATE' column and the target variable
y = df['CSUSHPINSA']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
model = xgb.XGBRegressor()

# Define the hyperparameters to tune
param_grid = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.05, 0.1, 0.2],
              'max_depth': [3, 4, 5],
              'subsample': [0.7, 0.8, 0.9]}

# Use GridSearchCV to search for the best hyperparameters
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding mean cross-validated score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Make predictions on the test data using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model using mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-Squared:", r2)
