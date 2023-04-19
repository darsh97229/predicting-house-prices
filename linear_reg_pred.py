import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the FRED house price indexes dataset
data = pd.read_csv('CSUSHPINSA.csv', parse_dates=True, index_col='DATE')

# Create a new column for the target variable (house price index)
data['HPI_target'] = data['CSUSHPINSA'].shift(-1)

# Drop rows with missing values
data.dropna(inplace=True)

# Split the data into training and test sets
X = data.drop('HPI_target', axis=1)
y = data['HPI_target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
model = LinearRegression()
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
