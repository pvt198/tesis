import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Cargar datos desde CSV
file_path = "datos_unidos.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)

# Convertir los valores numéricos correctamente
for col in ["Open", "High", "Low", "Close"]:
    df[col] = df[col].str.replace(",", "").astype(float)

# Plot de los datos
df.head()


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
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]

means_test = means[split:]
stds_test = stds[split:]

X_train = X_train.reshape(-1, input_steps, 1)
X_test = X_test.reshape(-1, input_steps, 1)


# Función para crear el modelo con diferentes capas densas
def create_model(neurons, learning_rate, dense_layers):
    model = Sequential([
        SimpleRNN(30, activation="tanh", return_sequences=False, input_shape=(input_steps, 1)),
    ])

    # Agregar capas densas según el número de capas especificado
    for i in range(dense_layers):
        model.add(Dense(neurons/(i+1), activation="relu"))  # "neurons/(i+1)" neuronas en cada capa densa

    model.add(Dense(output_steps, activation='linear')) # Capa de salida
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


# Prueba de diferentes hiperparámetros manualmente
param_grid = {
    "neurons": [10, 20, 50],
    "learning_rate": [0.001, 0.01],
    "batch_size": [8, 32, 64],
    "epochs": [10, 20, 30],
    "dense_layers": [1, 2, 3]
}

best_loss = float("inf")
best_params = None
best_model = None

for neurons in param_grid["neurons"]:
    for learning_rate in param_grid["learning_rate"]:
        for batch_size in param_grid["batch_size"]:
            for epochs in param_grid["epochs"]:
                for dense_layers in param_grid["dense_layers"]:
                    print(
                        f"\nProbando configuración: Neuronas={neurons}, LR={learning_rate}, Batch={batch_size}, Épocas={epochs}, Capas Densas={dense_layers}")

                    model = create_model(neurons, learning_rate, dense_layers)
                    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                                        validation_data=(X_test, Y_test))

                    loss = model.evaluate(X_test, Y_test, verbose=0)
                    print(f"Pérdida en prueba (MSE): {loss:.4f}")

                    if loss < best_loss:
                        best_loss = loss
                        best_params = (neurons, learning_rate, batch_size, epochs, dense_layers)
                        best_model = model

# Mostrar los mejores hiperparámetros encontrados
print("\nMejor configuración encontrada:")
print(
    f"Neuronas={best_params[0]}, Learning Rate={best_params[1]}, Batch Size={best_params[2]}, Épocas={best_params[3]}, Capas Densas={best_params[4]}")
print(f"Mejor pérdida en prueba: {best_loss:.4f}")

# Predicción y conversión a escala original
pred_scaled = best_model.predict(X_test)
pred_original = (pred_scaled * stds_test[:, None]) + means_test[:, None]
real_original = (Y_test * stds_test[:, None]) + means_test[:, None]

# Graficar predicciones vs. valores reales
plt.figure(figsize=(12, 6))
plt.plot(real_original[:, 0], label="Real", color="blue")  # Solo muestra el primer día de cada predicción
plt.plot(pred_original[:, 0], label="Predicción", color="red", linestyle="dashed")  # Predicción del primer día
plt.title("Predicción del Precio de Cierre de Nasdaq")
plt.xlabel("Días")
plt.ylabel("Precio")
plt.legend()
plt.show()
