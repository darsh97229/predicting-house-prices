# Prophet Prediction
Prophet is a time series forecasting model developed by Facebook. It is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.

In this code, we first prepare the time series data in the format required by Prophet. We then split the data into training and testing sets and instantiate the Prophet model.

After fitting the model on the training data, we create a future dataframe to hold the forecast values for the next 12 months. We then generate the forecast using the predict method of the model.

We can plot the forecast using the plot method of the model. Finally, we evaluate the performance of the model on the test set by calculating the RMSE between the forecasted values and the actual values.

To tune the Prophet model, we can adjust the hyperparameters of the model, such as the number of changepoints, the strength of the seasonality components, and the seasonality mode. We can also add custom holiday effects to the model to account for any specific events that may affect the time series.

This code will perform a grid search over the specified hyperparameters of the Prophet model and print the best RMSE and corresponding hyperparameters. We can then use these optimal hyperparameters to train and evaluate our final model. Note that this grid search may take some time to complete, especially if the parameter grid is large.



```
```