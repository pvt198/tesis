import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.callbacks import EarlyStopping

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Cargar datos desde CSV
file_path = "datos_unidos.csv"
df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
#print(df)

# Cargar los datos del sentimiento de mercado
sentiment_file_path = "./sentimiento_de_mercado/aggregated_sentiment_old.csv"
df_sentiment = pd.read_csv(sentiment_file_path, parse_dates=["Date"], dayfirst=True)
#print(df_sentiment)

# Merge ambos DataFrames basados en la columna 'Date'
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'], format='%Y-%m-%d')
df = pd.merge(df, df_sentiment[['Date', 'Average Weighted Sentiment']], on='Date', how='inner')
df = df.dropna()  # Elimina las filas con valores NaN
#print(df)

# Nome file donde salvar las predicciones
out_pred_name = "RNN_solo_Precio_y_Sentimiento"

# Definir los valores a explorar
input_steps = 30
output_steps = 1

param_grid = {
    "neurons": [10],#,20,30],
    "learning_rate": [0.0001],
    "batch_size": [32],
    "epochs": [1000],
    "dense_layers": [1],#,2,3],
    "RNN": [30],#,60],
    "recurrent": [0],#, 1],
    "use_sentiment": [False],  # Añadir la opción para usar o no el sentimiento
    "rnn_type": ['SimpleRNN', 'GRU', 'LSTM']  # Añadir la opción para elegir el tipo de celda recurrente
}

# Generar todas las combinaciones posibles de hiperparámetros
param_combinations = list(itertools.product(
    param_grid["neurons"],
    param_grid["learning_rate"],
    param_grid["batch_size"],
    param_grid["epochs"],
    param_grid["dense_layers"],
    param_grid["recurrent"],
    param_grid["RNN"],
    param_grid["use_sentiment"],
    param_grid["rnn_type"]  # Añadir la opción para elegir el tipo de celda RNN
))

# Convertir los valores numéricos correctamente
for col in ["Open", "High", "Low", "Close"]:
    df[col] = df[col].str.replace(",", "").astype(float)


# Función para crear secuencias para la RNN
def create_sequences(data, sentiment, input_steps=7, output_steps=1, use_sentiment=True):
    X, Y, means, stds = [], [], [], []
    for i in range(len(data) - input_steps - output_steps):
        # Sequences of both price (Close) and sentiment (Weighted Sentiment)
        if use_sentiment:
            seq = np.column_stack((data[i:i + input_steps], sentiment[i:i + input_steps]))
        else:
            seq = np.expand_dims(data[i:i + input_steps], axis=1)  # Solo precio, forma (input_steps, 1)

        mean = np.mean(seq, axis=0)
        std = np.std(seq, axis=0)

        normalized_seq = (seq - mean) / std

        X.append(normalized_seq)
        Y.append((data[i + input_steps:i + input_steps + output_steps] - mean[0]) / std[0])  # Normalizar solo el precio

        means.append(mean)
        stds.append(std)

    return np.array(X), np.array(Y), np.array(means), np.array(stds)


# Función para crear el modelo
def create_model(neurons, learning_rate, dense_layers, recurrent, recurrent_cells, use_sentiment=True, rnn_type='SimpleRNN'):
    model = Sequential()

    # Elegir la celda RNN según rnn_type
    if rnn_type == 'SimpleRNN':
        rnn_layer = SimpleRNN
    elif rnn_type == 'GRU':
        rnn_layer = tf.keras.layers.GRU
    elif rnn_type == 'LSTM':
        rnn_layer = tf.keras.layers.LSTM
    else:
        raise ValueError(f"Unknown rnn_type: {rnn_type}")

    model.add(rnn_layer(recurrent_cells, activation="tanh", return_sequences=recurrent,
                        input_shape=(
                        input_steps, 2 if use_sentiment else 1)))  # 2 características si usamos sentimiento

    # Agregar capas densas según el número de capas especificado
    for i in range(dense_layers):
        if recurrent == 0:
            model.add(Dense(int(neurons / (i + 1)), activation="relu"))
        else:
            if i == dense_layers - 1:
                model.add(rnn_layer(int(neurons / (i + 1)), activation="tanh", return_sequences=False))
            else:
                model.add(rnn_layer(int(neurons / (i + 1)), activation="tanh", return_sequences=True))

    if recurrent == 1:
        model.add(Dense(int(neurons / (dense_layers)), activation="relu"))
    model.add(Dense(output_steps, activation='linear'))  # Capa de salida
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    # Grafica el Modelo
    # plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)
    # img = plt.imread('./model_plot.png')
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    #model.summary()
    return model


# Inicializar una lista para almacenar los resultados
results = []

# Para parar el entrenamiento si no hay mejoramientos
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=100,
    min_delta=0.00001,
    restore_best_weights=True,
    verbose=1
)

# Loop sobre todas las combinaciones de hiperparámetros
for params in param_combinations:
    neurons, learning_rate, batch_size, epochs, dense_layers, recurrent, RNN, use_sentiment, rnn_type = params
    X = []
    Y = []
    # Preparar datos
    X, Y, means, stds = create_sequences(df["Close"].values, df["Average Weighted Sentiment"].values, input_steps,
                                         output_steps, use_sentiment)
    split = int(len(X) * 0.8)
    X_trainF, Y_trainF = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]
    means_test = means[split:]
    stds_test = stds[split:]
    split2 = int(len(X_trainF) * 0.8)
    X_train, Y_train = X_trainF[:split2], Y_trainF[:split2]
    X_val, Y_val = X_trainF[split2:], Y_trainF[split2:]
    # Aquí verificamos la forma de entrada
    X_train = X_train.reshape(-1, input_steps, 2 if use_sentiment else 1)
    X_val = X_val.reshape(-1, input_steps, 2 if use_sentiment else 1)
    X_test = X_test.reshape(-1, input_steps, 2 if use_sentiment else 1)
    print(
        f"Entrenando con: neurons={neurons}, learning_rate={learning_rate}, "
        f"batch_size={batch_size}, epochs={epochs}, dense_layers={dense_layers},"
        f" recurrent={recurrent}, RNN={RNN}, use_sentiment={use_sentiment}, rnn_type={rnn_type}")

    # Crear el modelo con los parámetros actuales
    model = create_model(neurons, learning_rate, dense_layers, recurrent, RNN, use_sentiment, rnn_type)

    # Entrenar el modelo
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                        validation_data=(X_val, Y_val), callbacks=[early_stopping])

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
        "val_loss": val_loss,
        "recurrent": recurrent,
        "RNN": RNN,
        "use_sentiment": use_sentiment,
        "rnn_type": rnn_type  # Guardar el tipo de celda RNN
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
    int(best_params["neurons"]),
    float(best_params["learning_rate"]),
    int(best_params["dense_layers"]),
    int(best_params["recurrent"]),
    int(best_params["RNN"]),
    best_params["use_sentiment"],
    best_params["rnn_type"]
)

# Guardar el modelo o usarlo para predicciones si es necesario
#best_model.save("best_model.h5")

# Train the model with the best parameters
history = best_model.fit(X_train, Y_train, epochs=best_params["epochs"], batch_size=best_params["batch_size"],
                         validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=1)

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()


# Make predictions on the test set

# Inverse normalization for predictions and actual values
predictions_rescaled = predictions * stds_test[:, 0].reshape(-1, 1) + means_test[:, 0].reshape(-1, 1)
Y_test_rescaled = Y_test * stds_test[:, 0].reshape(-1, 1) + means_test[:, 0].reshape(-1, 1)
# Extract the dates for the test set
test_dates = df['Date'].iloc[-len(Y_test_rescaled):].values  # Adjust the range to match the length of Y_test

# Plot actual vs predicted values on the test set
plt.figure(figsize=(12, 6))
plt.plot(test_dates, Y_test_rescaled, label='Actual Values', color='blue', alpha=0.6)
plt.plot(test_dates, predictions_rescaled, label='Predicted Values', color='red', alpha=0.6)
plt.title("Actual vs Predicted Test Values")
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.xticks(rotation=45)  # Rotate the dates to avoid overlap
plt.legend()
plt.tight_layout()
plt.show()


# Evaluate the test loss
test_loss = best_model.evaluate(X_test, Y_test)
print(f"Test Loss (MSE): {test_loss:.4f}")