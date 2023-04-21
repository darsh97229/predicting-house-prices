import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the FRED house price index dataset
data = pd.read_csv('../Data/CSUSHPINSA.csv', index_col='DATE', parse_dates=True)

# Split the data into training and testing sets
train_data = data.iloc[:-12, :]
test_data = data.iloc[-12:, :]

# Create the ARIMA model
model = ARIMA(train_data, order=(3,1,0))
model_fit = model.fit()

# Make predictions
train_predictions = model_fit.predict(start=train_data.index[1], end=train_data.index[-1])
test_predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

# Evaluate the model performance
mse = mean_squared_error(test_data, test_predictions)
rmse = np.sqrt(mse)
print("Test MSE: {:.3f}".format(mse))
print("Test RMSE: {:.3f}".format(rmse))

# Plot the predicted vs. actual house price indexes
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(test_data.index, test_predictions, label='Predicted')
plt.xlabel('Year')
plt.ylabel('House Price Index')
plt.title('FRED House Price Indexes')
plt.legend()
plt.show()


# Define the p, d, and q parameter ranges
p_range = range(0, 5)
d_range = range(0, 3)
q_range = range(0, 5)

# Generate all possible combinations of p, d, and q values
pdq_combinations = list(itertools.product(p_range, d_range, q_range))

# Define the parameters to evaluate the models with
eval_parameters = []

for pdq in pdq_combinations:
    try:
        # Fit the ARIMA model with the current pdq combination
        model = ARIMA(train_data, order=pdq)
        model_fit = model.fit()

        # Make predictions on the test set
        test_predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

        # Evaluate the model's performance with RMSE
        mse = mean_squared_error(test_data, test_predictions)
        rmse = np.sqrt(mse)

        # Append the evaluation parameters and RMSE to the list
        eval_parameters.append((pdq, rmse))
    except:
        # Ignore any errors and continue with the next combination
        continue

# Sort the evaluation parameters list by RMSE in ascending order
eval_parameters.sort(key=lambda x: x[1])

# Print the 5 best performing models with their respective evaluation parameters
for i in range(5):
    print("Evaluation Parameters: ", eval_parameters[i][0])
    print("RMSE: {:.3f}".format(eval_parameters[i][1]))

