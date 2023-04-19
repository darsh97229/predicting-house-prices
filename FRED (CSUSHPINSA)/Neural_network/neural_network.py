import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the FRED house price index dataset
data = pd.read_csv('../Data/CSUSHPINSA.csv', index_col='DATE', parse_dates=True)

# Preprocess the data
train_data = data.iloc[:-12, :]
test_data = data.iloc[-12:, :]
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Create the training and testing data
X_train = []
y_train = []
for i in range(12, len(train_scaled)):
    X_train.append(train_scaled[i-12:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = []
y_test = []
for i in range(12, len(test_scaled)):
    X_test.append(test_scaled[i-12:i, 0])
    y_test.append(test_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create the neural network model
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(units=1))
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform([y_train])
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

# Evaluate the model performance
mse = mean_squared_error(y_test[0], test_predictions[:, 0])
rmse = np.sqrt(mse)
print("Test MSE: {:.3f}".format(mse))
print("Test RMSE: {:.3f}".format(rmse))

# Plot the predicted vs. actual house price indexes
plt.plot(test_data.index, y_test[0], label='Actual')
plt.plot(test_data.index, test_predictions[:, 0], label='Predicted')
plt.xlabel('Year')
plt.ylabel('House Price Index')
plt.title('FRED House Price Indexes')
plt.legend()
plt.show()
