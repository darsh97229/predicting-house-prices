# KNN Prediction

Used scikit-learn's KNeighborsRegressor class to define and fit a KNN model to the FRED house price index dataset. We defined the number of neighbors (n_neighbors) to be 5, which is a common default value. We then used the model to make predictions on the test data and evaluated its performance using mean squared error and R-squared.

### For tuning:
Used scikit-learn's GridSearchCV class to perform a grid search over the hyperparameter 'n_neighbors' (the number of neighbors to use in the KNN model). We searched over values of 3, 5, 7, and 9, and used 5-fold cross validation to evaluate the performance of each hyperparameter combination. We then used the best hyperparameters to fit the model to the training data and make predictions on the test data. Finally, we evaluated the model's performance using mean squared error and R-squared.
