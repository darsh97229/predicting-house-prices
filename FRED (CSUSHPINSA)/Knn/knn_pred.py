import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
model = KNeighborsRegressor(n_neighbors=5)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-Squared:", r2)
