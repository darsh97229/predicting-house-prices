# Gradient Boosting Prediction
In this updated code, we use the GradientBoostingRegressor class from scikit-learn's ensemble module to define a gradient boosting regression model. We again standardize the training and test data using scikit-learn's StandardScaler class.

We define a parameter grid with a range of hyperparameters to test using a grid search with 5-fold cross-validation. We specify the GradientBoostingRegressor estimator as the estimator to use, and set the number of jobs to -1 to use all available CPU cores for parallel computation.

We fit the grid search object to the standardized training data using the fit() method, which will perform a search over the hyperparameter grid and return the best set of hyperparameters found. We print the best hyperparameters to the console.

Next, we fit a gradient boosting regression model to the standardized training data with the best hyperparameters found by the grid search, and make predictions on the standardized test data using the predict() method.

We evaluate the performance of the model using the mean squared error and root mean squared error metrics. Finally, we plot the predicted vs. actual house price indexes using matplotlib.

With gradient boosting regression, we can try to tune hyperparameters such as the learning rate, number of estimators, maximum depth of each tree, subsampling rate, and maximum number of features to consider when splitting a node. By tuning these hyperparameters, we can often improve the performance of the model and make more accurate predictions.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

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

# Standardize the training and test data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4],
    'subsample': [0.5, 0.8, 1.0],
    'max_features': ['sqrt', 'log2', None],
}

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_std, y_train)
print('Best hyperparameters:', grid_search.best_params_)

# Fit a gradient boosting regression model to the training data with the best hyperparameters
model = GradientBoostingRegressor(**grid_search.best_params_)
model.fit(X_train_std, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_std)

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