# Random Forest Prediction
In this code, we define a parameter grid with a range of hyperparameters to test using a grid search with 5-fold cross-validation. The GridSearchCV class from scikit-learn's model_selection module is used for this purpose. We specify the random forest regressor as the estimator to use, and set the number of jobs to -1 to use all available CPU cores for parallel computation.

We then fit the grid search object to the training data using the fit() method, which will perform a search over the hyperparameter grid and return the best set of hyperparameters found. We print the best hyperparameters to the console.

Next, we fit a random forest regression model to the training data with the best hyperparameters found by the grid search, and make predictions on the test data using the predict() method. We evaluate the performance of the model using the root mean squared error (RMSE) metric, and plot the predicted vs. actual house price indexes using the matplotlib library.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Load the FRED house price indexes dataset
data = pd.read_csv('../Data/CSUSHPINSA.csv', parse_dates=True, index_col='DATE')

# Create a new column for the target variable (house price index)
data['HPI_target'] = data['CSUSHPINSA'].shift(-1)

# Drop rows with missing values
data.dropna(inplace=True)

# Split the data into training and test sets
X = data.drop('HPI_target', axis=1)
y = data['HPI_target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)
print('Best hyperparameters:', grid_search.best_params_)

# Fit a random forest regression model to the training data with the best hyperparameters
model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Plot the predicted vs. actual house price indexes
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.show()
```