import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import itertools

# Cargar datos desde CSV
file_path = "./prices/datos_unidos.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)

# Nome file donde salvar las predicciones
out_pred_name = "RNN_solo_Precio"

# Convertir los valores numéricos correctamente
for col in ["Open", "High", "Low", "Close"]:
    df[col] = df[col].str.replace(",", "").astype(float)


# Crear secuencias para la RNN
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
input_steps = 30
output_steps = 7
X, Y, means, stds = create_sequences(df["Close"].values, input_steps, output_steps)

# Dividir en entrenamiento y prueba
split = int(len(X) * 0.8)
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


# Función para crear el modelo
def create_model(neurons, learning_rate, dense_layers):
    model = Sequential([
        SimpleRNN(30, activation="tanh", return_sequences=False, input_shape=(input_steps, 1)),
    ])

    # Agregar capas densas según el número de capas especificado
    for i in range(dense_layers):
        model.add(Dense(neurons / (i + 1), activation="relu"))  # "neurons/(i+1)" neuronas en cada capa densa

    model.add(Dense(output_steps, activation='linear'))  # Capa de salida
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


# Definir los valores a explorar
param_grid = {
    "neurons": [10, 20, 30],
    "learning_rate": [0.0001, 0.001],
    "batch_size": [8, 32, 64],
    "epochs": [50],
    "dense_layers": [1, 2, 3]
}

# Generar todas las combinaciones posibles de hiperparámetros
param_combinations = list(itertools.product(
    param_grid["neurons"],
    param_grid["learning_rate"],
    param_grid["batch_size"],
    param_grid["epochs"],
    param_grid["dense_layers"]
))

# Inicializar una lista para almacenar los resultados
results = []

# Loop sobre todas las combinaciones de hiperparámetros
for params in param_combinations:
    neurons, learning_rate, batch_size, epochs, dense_layers = params
    print(
        f"Entrenando con: neurons={neurons}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}, dense_layers={dense_layers}")

    # Crear el modelo con los parámetros actuales
    model = create_model(neurons, learning_rate, dense_layers)

    # Entrenar el modelo
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                        validation_data=(X_val, Y_val))

    # Evaluar el modelo
    val_loss = min(history.history['val_loss'])
    print(f"Pérdida de validación (MSE): {val_loss:.4f}")

    # Guardar el resultado para cada combinación de parámetros
    results.append({
        "neurons": neurons,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "dense_layers": dense_layers,
        "val_loss": val_loss
    })

# Convertir los resultados a un DataFrame
results_df = pd.DataFrame(results)

# Mostrar los resultados
print("Resultados del Hyperparameter Sweep:")
print(results_df)

# Seleccionar el mejor modelo (el que tenga la menor pérdida de validación)
best_params = results_df.loc[results_df['val_loss'].idxmin()]
print(f"Mejores parámetros: {best_params}")

# Crear el modelo con los mejores parámetros
best_model = create_model(
    best_params["neurons"],
    best_params["learning_rate"],
    best_params["dense_layers"]
)

# Entrenar el modelo con los mejores parámetros
history = best_model.fit(X_train, Y_train, epochs=best_params["epochs"], batch_size=best_params["batch_size"],
                         verbose=1, validation_data=(X_val, Y_val))

# Evaluar el modelo en el conjunto de prueba
test_loss = best_model.evaluate(X_test, Y_test, verbose=0)
print(f"Pérdida en prueba (MSE): {test_loss:.4f}")
