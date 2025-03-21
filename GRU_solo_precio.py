import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense  # Import GRU layer instead of LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import GlorotUniform, HeUniform

# Función para fijar la semilla en todos los lugares necesarios
def set_seed(seed=42):
    np.random.seed(seed)  # Fija la semilla para NumPy
    tf.random.set_seed(seed)  # Fija la semilla para TensorFlow
set_seed(42)

# Cargar datos desde CSV
file_path = "datos_unidos.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)

# Nome file donde salver las predicciones
out_pred_name = "GRU_solo_Precio"  # Updated output file name

# Convertir los valores numéricos correctamente
for col in ["Open", "High", "Low", "Close"]:
    df[col] = df[col].str.replace(",", "").astype(float)

# Crear secuencias para la GRU
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

# Definir los hiperparámetros
neurons = 10
learning_rate = 0.0001
batch_size = 32
epochs = 100
dense_layers = 1

# Crear el modelo con los hiperparámetros fijos utilizando GRU
def create_model(neurons, learning_rate, dense_layers):
    model = Sequential([
        GRU(30, activation="tanh", return_sequences=False,
            kernel_initializer=HeUniform(seed=42), input_shape=(input_steps, 1)),  # Replace LSTM with GRU
    ])

    # Agregar capas densas según el número de capas especificado
    for i in range(dense_layers):
        model.add(Dense(neurons / (i + 1), activation="relu",
                  kernel_initializer=HeUniform(seed=42)))  # "neurons/(i+1)" neuronas en cada capa densa

    model.add(Dense(output_steps, activation='linear', kernel_initializer=HeUniform(seed=42)))  # Capa de salida
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

# Crear el modelo y entrenarlo
model = create_model(neurons, learning_rate, dense_layers)

# Grafica el Modelo
plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)
img = plt.imread('./model_plot.png')
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_val, Y_val))

# Función para encontrar la mínima pérdida de validación
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

# Evaluar el modelo
loss = model.evaluate(X_test, Y_test, verbose=0)
print(f"Pérdida en prueba (MSE): {loss:.4f}")

# Predicción y conversión a escala original
pred_scaled = model.predict(X_test)
pred_original = (pred_scaled * stds_test[:, None]) + means_test[:, None]
real_original = (Y_test * stds_test[:, None]) + means_test[:, None]

# Crear las fechas para las predicciones hasta el 15 de marzo de 2025
num_predictions = len(real_original)
start_date = pd.Timestamp('2022-01-01')
end_date = pd.Timestamp('2025-03-14')

# Calcular la frecuencia necesaria para dividir el rango de fechas
pred_dates = pd.date_range(start=start_date, end=end_date, periods=num_predictions)

# Graficar predicciones vs. valores reales
plt.figure(figsize=(12, 6))
plt.plot(pred_dates, real_original[:, 0], label="Real", color="blue")  # Valores reales
plt.plot(pred_dates, pred_original[:, 0], label="Predicción", color="red", linestyle="dashed")  # Predicciones
plt.title("Predicción del Precio de Cierre de Nasdaq")
plt.xlabel("Año")
plt.ylabel("Precio")
plt.xticks(rotation=90)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.legend()
plt.tight_layout()
plt.show()

df_results = pd.DataFrame({
    "Date": pred_dates.strftime('%Y-%m-%d'),  # Fechas de predicción
    "Actual_Close":  real_original[:, 0],  # Valores reales
    "Predicted_Close": pred_original[:, 0]  # Valores predichos
})

# Guardar el DataFrame en un archivo CSV
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
