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
