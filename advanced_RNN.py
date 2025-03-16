import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data from CSV
file_path = "NASDAQ_price_plus_macro.csv"  # Adjust the path to your CSV file
df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)

# Create sequences for RNN
def create_sequences(data, input_steps=30, output_steps=7):
    X, Y, means, stds = [], [], [], []
    for i in range(len(data) - input_steps - output_steps):

        seq = data[i:i + input_steps]
        # Normalize only the closing price
        price_seq = seq[:, 0]  # Get the 'Close' column
        mean = np.mean(price_seq)
        std = np.std(price_seq)

        # Normalize the closing price
        normalized_price_seq = (price_seq - mean) / std

        # Keep the other features unchanged
        normalized_seq = seq.copy()
        normalized_seq[:, 0] = normalized_price_seq  # Replace the normalized price in the sequence

        X.append(normalized_seq)
        Y.append((data[i + input_steps:i + input_steps + output_steps, 0] - mean) / std)  # Normalize the output as well

        means.append(mean)
        stds.append(std)

    return np.array(X), np.array(Y), np.array(means), np.array(stds)


# Prepare data
input_steps = 60
output_steps = 60
data_columns = ['Close', 'GDP', 'Unemployment Rate', 'Interest Rate', 'M2 Money Supply', 'Inflation']
data_values = df[data_columns].values  # Extract relevant columns
# Create sequences
X, Y, means, stds = create_sequences(data_values, input_steps, output_steps)
print(X)
print(np.shape(Y))

# Split the data into training and testing sets
split = int(len(X) * 0.8)
X_train, Y_train = X[:split], Y[:split]
split2 = int(len(X) * 0.8)
X_train, Y_train = X_train[:split2], Y_train[:split2]
X_val, Y_val =  X_train[split2:], Y_train[split2:]
X_test, Y_test = X[split:], Y[split:]
means_test = means[split:]
stds_test = stds[split:]

# Reshape X for RNN input
X_train = X_train.reshape(-1, input_steps, len(data_columns))
X_test = X_test.reshape(-1, input_steps, len(data_columns))

# Define hyperparameters
neurons = 20
learning_rate = 0.0001
batch_size = 8
epochs = 20
dense_layers = 3


# Create model
def create_model(neurons, learning_rate, dense_layers):
    model = Sequential([
        SimpleRNN(30, activation="tanh", return_sequences=False, input_shape=(input_steps, len(data_columns))),
    ])

    for i in range(dense_layers):
        model.add(Dense(neurons / (i + 1), activation="relu"))  # Add dense layers

    model.add(Dense(output_steps, activation='linear'))  # Output layer for predicting only the Close price
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


# Create and train the model
model = create_model(neurons, learning_rate, dense_layers)
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_val, Y_val))

# Evaluate the model
loss = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test Loss (MSE): {loss:.4f}")

# Make predictions
pred_scaled = model.predict(X_test)

pred_original = (pred_scaled * stds_test[:, None]) + means_test[:, None]  # Rescale predictions back to original
print(np.shape(pred_original))

# Create dates for predictions
pred_dates = df["Date"].values[split + input_steps:split + input_steps + len(pred_original)]

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(pred_dates, pred_original[:,0], label="Predicted Close Price", color="red", linestyle="dashed")
plt.plot(pred_dates, df["Close"].values[split + input_steps:split + input_steps + len(pred_original)], label="Actual Close Price", color="blue")  # Actual Close Price
plt.title("Predicted vs Actual Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Show only the year
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

df_results = pd.DataFrame({
    "Date": pred_dates,  # Fechas de predicción
    "Actual_Close": df["Close"].values[split + input_steps:split + input_steps + len(pred_original)],  # Valores reales
    "Predicted_Close": pred_original[:, 0]  # Valores predichos
})

# Guardar el DataFrame en un archivo CSV
df_results.to_csv("TEST_predictions_RNN_price_plus_macro.csv", index=False)



# Get the last date from the dataset
pred_dates = pd.to_datetime(pred_dates)  # Convert to datetime (if it's a NumPy array)
# Get the last date
last_date = pred_dates[-1]  # Use indexing instead of .iloc[-1]
# Generate the next 60 days
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=60, freq='D')
# Plot Proximo 60 dias
plt.figure(figsize=(12, 6))
plt.plot(np.concatenate((pred_dates, future_dates)), df["Close"].values[split + input_steps:split + input_steps + len(pred_original) + len(future_dates)], label="Actual Close Price", color="blue")  # Actual Close Price
plt.plot(pred_dates, pred_original[:,0], label="Predicted Close Price", color="red", linestyle="dashed")
plt.plot( future_dates,pred_original[-1,:], label="Predicted Close Price", color="red")
plt.title("Proximo 60 dias")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Show only the year
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()