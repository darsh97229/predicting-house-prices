# Random Forest Prediction
Prophet is a time series forecasting model developed by Facebook. It is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

In this code, we first prepare the time series data in the format required by Prophet. We then split the data into training and testing sets and instantiate the Prophet model.

After fitting the model on the training data, we create a future dataframe to hold the forecast values for the next 12 months. We then generate the forecast using the predict method of the model.

We can plot the forecast using the plot method of the model. Finally, we evaluate the performance of the model on the test set by calculating the RMSE between the forecasted values and the actual values.

To tune the Prophet model, we can adjust the hyperparameters of the model, such as the number of changepoints, the strength of the seasonality components, and the seasonality mode. We can also add custom holiday effects to the model to account for any specific events that may affect the time series.

This code will perform a grid search over the specified hyperparameters of the Prophet model and print the best RMSE and corresponding hyperparameters. We can then use these optimal hyperparameters to train and evaluate our final model. Note that this grid search may take some time to complete, especially if the parameter grid is large.



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