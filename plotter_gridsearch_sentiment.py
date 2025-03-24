import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datos en formato de diccionario (puedes cargarlos desde un archivo CSV también)
data = {
    "neurons": [10]*36 + [30]*36 + [60]*36,
    "learning_rate": [0.0001]*108,
    "batch_size": [32]*108,
    "epochs": [1000]*108,
    "dense_layers": [1]*12 + [2]*12 + [3]*12 + [1]*12 + [2]*12 + [3]*12 + [1]*12 + [2]*12 + [3]*12,
    "val_loss": [
        0.608982, 0.536916, 0.593816, 0.659654, 0.604424, 0.669197, 0.764584, 0.578134, 0.762979, 0.731220, 0.650821, 0.575953,
        0.661108, 0.519339, 0.522872, 0.902352, 0.498889, 0.727046, 0.607812, 0.631666, 0.737713, 0.873694, 0.667804, 0.664310,
        0.661035, 0.634247, 0.642264, 0.610660, 0.623133, 1.704108, 1.746235, 1.720406, 0.611147, 0.737450, 0.708541, 0.669715,
        0.635461, 0.533847, 0.527005, 0.672016, 0.580473, 0.638743, 0.617373, 0.746263, 0.525081, 0.813307, 0.647540, 0.662868,
        0.806332, 0.600357, 0.511365, 0.797492, 0.622322, 0.666889, 0.623564, 0.895791, 0.694854, 0.931702, 0.582135, 0.628952,
        0.727872, 0.574081, 0.559101, 0.567027, 0.613087, 0.637913, 0.608447, 0.576287, 0.712911, 0.779076, 0.625908, 0.727004,
        0.624463, 0.639378, 0.541739, 0.830399, 0.589346, 0.666736, 0.703517, 0.538224, 0.570182, 0.864886, 0.598979, 0.795912,
        0.685369, 0.598416, 0.492440, 0.665083, 0.579461, 0.634841, 0.553778, 0.515889, 0.701238, 0.901127, 0.577764, 0.621742,
        0.598754, 0.603879, 0.505419, 0.632030, 0.547911, 0.623517, 0.797661, 0.562848, 0.741685, 0.795035, 0.632310, 0.655205
    ],
    "recurrent": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 9,
    "celdasR": [30]*108,
    "use_sentiment": [False, False, False, True, True, True] * 18,
    "rnn_type": ["SimpleRNN", "GRU", "LSTM"] * 36
}

# Crear DataFrame
df = pd.DataFrame(data)

# Mostrar las primeras filas
print(df.head())

# Gráfica de comparación de val_loss por tipo de RNN
plt.figure(figsize=(10, 6))
sns.boxplot(x="rnn_type", y="val_loss", data=df)
plt.title("Comparación de pérdida de validación por tipo de RNN")
plt.xlabel("Tipo de RNN")
plt.ylabel("Pérdida de validación (val_loss)")
plt.show()

# Promedio de val_loss por tipo de RNN
print(df.groupby("rnn_type")["val_loss"].mean())

# Gráfica de boxplot de val_loss por tipo de RNN y uso de sentimiento
plt.figure(figsize=(12, 6))
sns.boxplot(x="rnn_type", y="val_loss", hue="use_sentiment", data=df, palette="Set1")
plt.title("Distribución de la pérdida de validación por tipo de RNN y uso de sentimiento")
plt.xlabel("Tipo de RNN")
plt.ylabel("Pérdida de validación (val_loss)")
plt.legend(title="Uso de Sentimiento", loc="upper right")
plt.show()

# Gráfica de boxplot de val_loss por tipo de RNN, uso de sentimiento y recurrente
plt.figure(figsize=(14, 6))
sns.boxplot(x="rnn_type", y="val_loss", hue="dense_layers", data=df, palette="Set1")
sns.stripplot(x="rnn_type", y="val_loss", hue="dense_layers", data=df, palette="Set1", dodge=True, marker="o", alpha=0.5)
# Ajustar títulos y etiquetas
plt.title("Distribución de la pérdida de validación por tipo de RNN, por capas ocultas recurrentes o no")
plt.xlabel("Tipo de RNN")
plt.ylabel("Pérdida de validación (val_loss)")
plt.legend(title="Capas Ocultas Recurrentes", loc="upper right")
plt.show()
