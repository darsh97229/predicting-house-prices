import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the FRED house price index dataset
df = pd.read_csv('../Data/CSUSHPINSA.csv')

# Split the data into features (X) and target variable (y)
X = df.drop(['DATE', 'CSUSHPINSA'], axis=1) # Drop the 'DATE' column and the target variable
y = df['CSUSHPINSA']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the KNN model
knn = KNeighborsRegressor()

# Define the hyperparameters to search over
param_grid = {'n_neighbors': [3, 5, 7, 9]}

# Use GridSearchCV to find the best hyperparameters
grid = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# Fit the model to the training data using the best hyperparameters
model = grid.best_estimator_
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid.best_params_)
print("Mean Squared Error:", mse)
print("R-Squared:", r2)
