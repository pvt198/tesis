import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

input_steps = 30
output_steps = 7

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

# Parámetros
neurons = 20
learning_rate = 0.01

# Crear modelos con 1, 2 y 3 capas densas
dense_layer_counts = [1, 2, 3]
models = []
for layers in dense_layer_counts:
    model = create_model(neurons, learning_rate, layers)
    models.append(model)

# Guardar las arquitecturas en imágenes
for layers, model in zip(dense_layer_counts, models):
    plot_model(model, to_file=f"rnn_architecture_{layers}_layers.png", show_shapes=True, show_layer_names=True)

# Mostrar las imágenes generadas
plt.figure(figsize=(15, 5))

for i, layers in enumerate(dense_layer_counts):
    img = plt.imread(f"rnn_architecture_{layers}_layers.png")
    plt.subplot(1, 3, i + 1)  # Crear 1 fila y 3 columnas
    plt.imshow(img)
    plt.axis("off")  # Quitar ejes

plt.tight_layout()
plt.show()
