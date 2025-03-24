import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import random, os

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Cargar datos desde CSV
file_path = "datos_unidos.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)

# Definir los valores a explorar
input_steps = 30
output_steps = 1

param_grid = {
    "neurons": [10, 20, 30],
    "learning_rate": [0.0001],
    "batch_size": [32],
    "epochs": [1000],
    "dense_layers": [1, 2, 3],
    "LSTM_units": [30],
    "recurrent": [0]
}

# Generar todas las combinaciones posibles de hiperparámetros
param_combinations = list(itertools.product(
    param_grid["neurons"],
    param_grid["learning_rate"],
    param_grid["batch_size"],
    param_grid["epochs"],
    param_grid["dense_layers"],
    param_grid["recurrent"],
    param_grid["LSTM_units"]
))

# Convertir los valores numéricos correctamente
for col in ["Open", "High", "Low", "Close"]:
    df[col] = df[col].str.replace(",", "").astype(float)


# Crear secuencias para la LSTM
def create_sequences(data, input_steps=30, output_steps=7):
    X, Y, means, stds = [], [], [], []
    for i in range(len(data) - input_steps - output_steps):
        seq = data[i:i + input_steps]
        mean = np.mean(seq)
        std = np.std(seq)
        normalized_seq = (seq - mean) / std
        X.append(normalized_seq)
        Y.append((data[i + input_steps:i + input_steps + output_steps] - mean) / std)
        means.append(mean)
        stds.append(std)
    return np.array(X), np.array(Y), np.array(means), np.array(stds)


# Preparar datos
X, Y, means, stds = create_sequences(df["Close"].values, input_steps, output_steps)

# Dividir en entrenamiento y prueba
split = int(len(X) * 0.9105)
X_trainF, Y_trainF = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]
means_test = means[split:]
stds_test = stds[split:]

split2 = int(len(X_trainF) * 0.8)
X_train, Y_train = X_trainF[:split2], Y_trainF[:split2]
X_val, Y_val = X_trainF[split2:], Y_trainF[split2:]

X_train = X_train.reshape(-1, input_steps, 1)
X_val = X_val.reshape(-1, input_steps, 1)
X_test = X_test.reshape(-1, input_steps, 1)


# Función para crear el modelo con LSTM
def create_model(neurons, learning_rate, dense_layers, recurrent, lstm_units):
    model = Sequential([
        LSTM(lstm_units, activation="tanh", return_sequences=bool(recurrent), input_shape=(input_steps, 1)),
    ])

    for i in range(dense_layers):
        model.add(Dense(int(neurons / (i + 1)), activation="relu"))

    model.add(Dense(output_steps, activation='linear'))  # Capa de salida
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


# Inicializar lista para almacenar resultados
results = []

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.0001,
    restore_best_weights=True,
    verbose=1
)

# Loop sobre combinaciones de hiperparámetros
for params in param_combinations:
    neurons, learning_rate, batch_size, epochs, dense_layers, recurrent, lstm_units = params
    print(
        f"Entrenando con: neurons={neurons}, learning_rate={learning_rate}, "
        f"batch_size={batch_size}, epochs={epochs}, dense_layers={dense_layers}, "
        f"recurrent={recurrent}, LSTM_units={lstm_units}")

    model = create_model(neurons, learning_rate, dense_layers, recurrent, lstm_units)

    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping]
    )

    val_loss = min(history.history['val_loss'])
    print(f"Pérdida de validación (MSE): {val_loss:.4f}")

    results.append({
        "neurons": neurons,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "dense_layers": dense_layers,
        "val_loss": val_loss,
        "recurrent": recurrent,
        "LSTM_units": lstm_units
    })

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)
print("Resultados del Hyperparameter Sweep:")
print(df_results)

# Seleccionar el mejor modelo
best_params = df_results.loc[df_results['val_loss'].idxmin()]
print(f"Mejores parámetros:\n {best_params}")

# Crear el mejor modelo
best_model = create_model(
    int(best_params["neurons"]),
    float(best_params["learning_rate"]),
    int(best_params["dense_layers"]),
    int(best_params["recurrent"]),
    int(best_params["LSTM_units"])
)
