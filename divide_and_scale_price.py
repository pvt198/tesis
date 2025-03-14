import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Cargar el archivo datos_unidos.csv
df = pd.read_csv('datos_unidos.csv')

# Verificar si las columnas necesarias existen
cols = ['Open', 'Close', 'High', 'Low']
if not set(cols).issubset(df.columns):
    raise ValueError("El archivo CSV no contiene todas las columnas necesarias: 'Open', 'Close', 'High', 'Low'")

# Convertir las columnas a valores numéricos (por si tienen comas)
df[cols] = df[cols].replace(',', '', regex=True).astype(float)

# Lista para almacenar los segmentos
segmentos = []

# Iterar sobre el DataFrame para extraer secciones de 30 días y los siguientes 7 días
for i in range(len(df) - 30 - 7):
    # Extraer los 30 días de entrada y los 7 días de salida
    entrada_30_dias = df.iloc[i:i+30][cols].values  # Convertir a numpy array
    salida_7_dias = df.iloc[i+30:i+37][cols].values  # Convertir a numpy array

    # Normalizar usando StandardScaler basado en los 30 días
    scaler = StandardScaler()
    entrada_normalizada = scaler.fit_transform(entrada_30_dias)  # Ajusta y transforma los 30 días
    salida_normalizada = scaler.transform(salida_7_dias)  # Usa la misma normalización en los 7 días

    # Guardar el segmento en la lista
    segmentos.append({
        'input': entrada_normalizada,
        'output': salida_normalizada,
        'input_original': entrada_30_dias,
        'output_original': salida_7_dias
    })

# Seleccionar un batch (ejemplo el primero)
batch = segmentos[0]
input_batch = batch['input']
output_batch = batch['output']
input_original = batch['input_original']
output_original = batch['output_original']

# Crear un índice de tiempo ficticio para el eje x
dias_input = np.arange(1, 31)  # Días 1-30 (entrada)
dias_output = np.arange(31, 38)  # Días 31-37 (salida)

# ---------------------------- #
# 📌 Gráfico 1: Datos originales #
# ---------------------------- #
plt.figure(figsize=(12, 6))

for i, col in enumerate(cols):
    plt.plot(dias_input, input_original[:, i], label=f"{col} (Input)", linestyle='solid')
    plt.plot(dias_output, output_original[:, i], label=f"{col} (Output)", linestyle='dashed')

plt.title('Datos Originales: 30 días de entrada y sus 7 días asociados')
plt.xlabel('Días')
plt.ylabel('Valor Real')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------- #
# 📌 Gráfico 2: Datos Normalizados #
# ---------------------------- #
plt.figure(figsize=(12, 6))

for i, col in enumerate(cols):
    plt.plot(dias_input, input_batch[:, i], label=f"{col} (Input)", linestyle='solid')
    plt.plot(dias_output, output_batch[:, i], label=f"{col} (Output)", linestyle='dashed')

plt.title('Datos Normalizados: 30 días de entrada y sus 7 días asociados')
plt.xlabel('Días')
plt.ylabel('Valores Normalizados')
plt.legend()
plt.grid(True)
plt.show()
