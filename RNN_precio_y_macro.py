import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.initializers import GlorotUniform, HeUniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LayerNormalization
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def create_sequences(data, input_steps=30, output_steps=7):
    X, Y, means, stds = [], [], [], []

    for i in range(len(data) - input_steps - output_steps):
        seq = data[i:i + input_steps]
        future_seq = data[i + input_steps:i + input_steps + output_steps]
        mean = np.mean(seq, axis=0)
        std = np.std(seq, axis=0)
        std[std == 0] = 1
        normalized_seq = (seq - mean) / std
        normalized_future_seq = (future_seq - mean) / std
        X.append(normalized_seq)
        Y.append(normalized_future_seq[:, 0])
        means.append(mean[0])
        stds.append(std[0])

    return np.array(X), np.array(Y), np.array(means), np.array(stds)


# Hiperparametros
input_steps = 360
output_steps = 7
neurons = 30
learning_rate = 0.0001
batch_size = 32
epochs = 30
dense_layers = 1
RRN_cell = 60
advanced = True

#  Cargar datos desde CSV
file_path = "results/NASDAQ_price_plus_macro.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
data_columns = ['Close']#, 'GDP', 'Unemployment Rate', 'Interest Rate', 'M2 Money Supply', 'Inflation']
data_values = df[data_columns].values
X, Y, means, stds = create_sequences(data_values, input_steps, output_steps)

#Nome file donde salver las predicciones
out_pred_name = "RNN_Precio_y_macro"

# Dividir en entrenamiento y prueba
split = int(len(X) * 0.9105)
X_trainF, Y_trainF = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]
means_test = means[split:]
stds_test = stds[split:]

#debug plot
print(np.shape(X_trainF))
X_trainF = np.clip(X_trainF, -4, 4)
min_vals = np.min(X_trainF, axis=(0, 1))
max_vals = np.max(X_trainF, axis=(0, 1))
mean_vals = np.mean(X_trainF, axis=(0, 1))
std_vals = np.std(X_trainF, axis=(0, 1))
#print("Min:", min_vals)
#print("Max:", max_vals)
#print("Mean:", mean_vals)
#print("Std:", std_vals)
flattened_data = X_trainF.reshape(-1, X_trainF.shape[-1])
print(np.shape(flattened_data))

# fig, axes = plt.subplots(1, 2, figsize=(15, 8))
# for i, ax in enumerate(axes.flat):
#     ax.hist(flattened_data[:, i], bins=50, alpha=0.7, color='b', edgecolor='black')
#     ax.set_title(data_columns[i])
#     ax.set_xlabel('Valor')
#     ax.set_ylabel('Frecuencia')
# plt.legend()
# plt.tight_layout()
# plt.show()

split2 = int(len(X_trainF) * 0.8)
X_train, Y_train = X_trainF[:split2], Y_trainF[:split2]
X_val, Y_val = X_trainF[split2:], Y_trainF[split2:]

X_train = X_train.reshape(-1, input_steps, len(data_columns))
X_val = X_val.reshape(-1, input_steps, len(data_columns))
X_test = X_test.reshape(-1, input_steps, len(data_columns))


def create_model(neurons, learning_rate, dense_layers):

    if advanced:
        model = Sequential([
            LSTM(RRN_cell, activation="tanh", return_sequences=False, input_shape=(input_steps, len(data_columns)))])
        LSTM(RRN_cell, activation="tanh", return_sequences=False,)
        model.add(Dense(neurons , activation="relu"))
        model.add(Dense(neurons , activation="relu"))
        model.add(Dense(output_steps, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return model

    else:
        model = Sequential([
            SimpleRNN(RRN_cell, activation="tanh", return_sequences=False, input_shape=(input_steps, len(data_columns)))])
        for i in range(dense_layers):
            model.add(Dense(neurons / (i + 1), activation="relu"))  # Add dense layers
        model.add(Dense(output_steps, activation='linear'))  # Output layer for predicting only the Close price
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return model


model = create_model(neurons, learning_rate, dense_layers)

# Grafica el Modelo
plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)
img = plt.imread('./model_plot.png')
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
model.summary()

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_val, Y_val))
loss = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test Loss (MSE): {loss:.4f}")

def find_min_val_loss(history):
    min_val_loss = min(history.history['val_loss'])
    min_epoch = history.history['val_loss'].index(min_val_loss) + 1  # +1 porque los índices comienzan en 0
    print(f"Mínima pérdida de validación (MSE): {min_val_loss:.4f} en la época {min_epoch}")
    return min_val_loss, min_epoch
min_val_loss, min_epoch = find_min_val_loss(history)

# Guardar historial de pérdidas en un CSV
loss_df = pd.DataFrame({
    "Epoch": np.arange(1, epochs + 1),
    "Loss": history.history['loss'],
    "Val_Loss": history.history['val_loss']
})
loss_file_path = "./results/"+out_pred_name+"_val_loss_history.csv"
loss_df.to_csv(loss_file_path, index=False)


# Prediciones
pred_scaled = model.predict(X_test)
pred_original = (pred_scaled * stds_test[:, None]) + means_test[:, None]

# Grafico Predicciones
#pred_dates = df["Date"].values[split + input_steps:split + input_steps + len(pred_original)]
#pred_dates = pd.to_datetime(pred_dates)
# Crear las fechas para las predicciones hasta el 15 de marzo de 2025
num_predictions = len(pred_original)
start_date = pd.Timestamp('2022-01-01')
end_date = pd.Timestamp('2025-03-14')
# Calcular la frecuencia necesaria para dividir el rango de fechas
pred_dates = pd.date_range(start=start_date, end=end_date, periods=num_predictions)

plt.figure(figsize=(12, 6))
plt.plot(pred_dates, pred_original[:,0], label="Predicted Close Price", color="red", linestyle="dashed")
plt.plot(pred_dates, df["Close"].values[split + input_steps:split + input_steps + len(pred_original)], label="Actual Close Price", color="blue")  # Actual Close Price
plt.title("Predicted vs Actual Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

#Guardar los Resultados
df_results = pd.DataFrame({
    "Date": pred_dates,  # Fechas de predicción
    "Actual_Close": df["Close"].values[split + input_steps:split + input_steps + len(pred_original)],  # Valores reales
    "Predicted_Close": pred_original[:, 0]  # Valores predichos
})
df_results.to_csv("./results/"+out_pred_name+"_predictions.csv", index=False)

# Graficar la pérdida de entrenamiento y validación
print(history.history.keys())
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento', color='blue')
plt.plot(history.history['val_loss'], label='Pérdida de validación', color='red', linestyle='dashed')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.title('Evolución de la Pérdida durante el Entrenamiento')
plt.legend()
plt.grid()
plt.show()
