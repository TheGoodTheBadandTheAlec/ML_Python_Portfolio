import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import keras.backend as K

# Set Jupyter Notebook to show the full printed output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Load the input data
input_csv = r'C:\Users\alecj\python\Crypto\ml_data_with_ta_simplified.csv'
output_csv = r'C:\Users\alecj\python\Crypto\ml_data_analysis.csv'
df = pd.read_csv(input_csv)

# Define the columns to be used for feature engineering
numeric_features = ['close', 'Volume USDT', 'Open', 'High', 'Low', 'tradecount', 'Short_MA', 'Price_Change','Long_MA','Short_EMA','Long_EMA','Short_RSI', 'Long_RSI','Short_MACD', 'Signal_Line','Long_MACD','Long_Signal_Line',
                    'Upper_Band','Lower_Band']

# Feature Engineering
lag_periods = [1, 2, 3, 4]
rolling_windows = [3, 7, 14]

# Create a function to shift the timestamps by 24 hours (86400 seconds)
def shift_time(df, hours=24):
    df['time'] += hours * 3600
    return df

# Shift timestamps by 24 hours
df = shift_time(df, hours=24)

for period in lag_periods:
    df.loc[:, f'close_lag_{period}'] = df.groupby('symbol')['close'].shift(period)

for window in rolling_windows:
    df.loc[:, f'rolling_mean_{window}'] = df.groupby('symbol')['close'].rolling(window=window).mean().values
    df.loc[:, f'rolling_std_{window}'] = df.groupby('symbol')['close'].rolling(window=window).std().values

# Fill missing values using forward-fill and backward-fill
df = df.groupby('symbol').apply(lambda group: group.ffill().bfill())

# Prepare the data for prediction
X = df[numeric_features + [f'close_lag_{period}' for period in lag_periods] +
       [f'rolling_mean_{window}' for window in rolling_windows] + [f'rolling_std_{window}' for window in rolling_windows]
       ]

# Scale numerical features
scaler = StandardScaler()

# Fit the scaler on X_numeric
X_numeric = X[numeric_features]
scaler.fit(X_numeric)

# Transform the data using the fitted scaler
X.loc[:, numeric_features] = scaler.transform(X_numeric)

# Create the LSTM models and predict for each symbol
symbols = df['symbol'].unique()
predicted_closes = []

def custom_loss(y_true, y_pred):
    # Calculate MAPE
    absolute_percentage_error = K.abs((y_true - y_pred) / y_true)
    mape = K.mean(absolute_percentage_error)

    # Calculate DA
    correct_direction = K.equal(K.sign(y_true[1:] - y_true[:-1]), K.sign(y_pred[1:] - y_pred[:-1]))
    da = K.mean(K.cast(correct_direction, dtype='float32'))

    # Combine MAPE and DA with a penalty for incorrect direction
    combined_loss = mape + (1 - da)

    return combined_loss

for symbol in symbols:
    # Extract data for the specific symbol
    symbol_data = df[df['symbol'] == symbol].copy()

    # Create input features for prediction
    symbol_features = symbol_data[numeric_features + [f'close_lag_{period}' for period in lag_periods] +
                                  [f'rolling_mean_{window}' for window in rolling_windows] +
                                  [f'rolling_std_{window}' for window in rolling_windows]]

    # Scale numerical features for the input features of the LSTM
    symbol_numeric_features = symbol_features[numeric_features]

    # Create a deep copy of the DataFrame to avoid the warning
    symbol_features = symbol_features.copy(deep=True)

    symbol_features.loc[:, numeric_features] = scaler.transform(symbol_numeric_features)

    # Reshape input data for LSTM
    input_shape = (X.shape[1], 1)
    symbol_features = symbol_features.values.reshape(symbol_features.shape[0], *input_shape)

    # Create the LSTM model with L2 regularization
    model = Sequential()
    model.add(LSTM(16, input_shape=input_shape, kernel_regularizer=l2(0.01), activation='relu'))
    model.add(Dense(1, activation='linear', kernel_regularizer=l2(0.01)))

    # Compile and train the model for the specific symbol with early stopping
    model.compile(optimizer='adam', loss=custom_loss)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(symbol_features, symbol_data['close'], epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    # Predict the close price for the specific symbol
    predicted_close = model.predict(symbol_features)
    predicted_closes.extend(predicted_close)

# Add the predicted closes to the original DataFrame
df['predicted_close'] = predicted_closes

# Save the DataFrame with predicted closes to a new CSV
df.to_csv(output_csv, index=False)

# Plot the predicted close against the real close for each symbol
for symbol in symbols:
    symbol_data = df[df['symbol'] == symbol]
    plt.figure(figsize=(10, 6))
    plt.plot(symbol_data['time'], symbol_data['close'], label='Real Close')
    plt.plot(symbol_data['time'], symbol_data['predicted_close'], label='Predicted Close', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.title(f'{symbol} - Real vs. Predicted Close')
    plt.legend()
    plt.grid()
    plt.show()
