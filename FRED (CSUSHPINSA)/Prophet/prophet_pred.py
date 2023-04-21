import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the dataset
df = pd.read_csv('../Data/CSUSHPINSA_prophet.csv')
#df.index = pd.to_datetime(df.index)

# print(df)

# Prepare the dataset
#df = df.reset_index().rename(columns={'DATE': 'ds', 'SP500': 'y'})
#df = df.fillna(method='ffill')

# Split the dataset into train and test sets
train_size = int(len(df) * 0.8)
train_df, test_df = df[:train_size], df[train_size:]

# Define the parameter grid to search over
param_grid = {
    'seasonality_mode': ['multiplicative', 'additive'],
    'changepoint_prior_scale': [0.01, 0.1, 1.0],
    'seasonality_prior_scale': [0.01, 0.1, 1.0]
}
# print(train_df)

# Grid search over the parameter grid
best_rmse = float('inf')
for mode in param_grid['seasonality_mode']:
    for prior_scale in param_grid['changepoint_prior_scale']:
        for seasonality_scale in param_grid['seasonality_prior_scale']:
            # Train the model
            model = Prophet(seasonality_mode=mode,
                            changepoint_prior_scale=prior_scale,
                            seasonality_prior_scale=seasonality_scale)
            model.fit(train_df)

            # Make predictions on the test set
            future = model.make_future_dataframe(periods=len(test_df))
            forecast = model.predict(future)
            fig = model.plot(forecast)
            fig.show()
            predictions = forecast['yhat'][-len(test_df):]

            # Evaluate the model
            rmse = sqrt(mean_squared_error(test_df['y'], predictions))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'seasonality_mode': mode,
                               'changepoint_prior_scale': prior_scale,
                               'seasonality_prior_scale': seasonality_scale}

print('Best RMSE: %.3f' % best_rmse)
print('Best parameters:', best_params)
