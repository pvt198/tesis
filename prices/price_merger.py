import pandas as pd
import os
import matplotlib.pyplot as plt

# Directorio donde se almacenan los archivos CSV
directorio = './'

# Lista para almacenar los DataFrames
dfs = []

# Iterar sobre los años del 2020 al 2026
for año in range(1990, 2026):
    # Crear el nombre del archivo CSV
    nombre_archivo = f"{año}.csv"
    ruta_archivo = os.path.join(directorio, nombre_archivo)

    # Verificar si el archivo existe
    if os.path.exists(ruta_archivo):
        # Leer el archivo CSV
        df = pd.read_csv(ruta_archivo)

        # Invertir el orden de las filas
        df = df[::-1].reset_index(drop=True)

        # Agregar el DataFrame invertido a la lista
        dfs.append(df)
    else:
        print(f"El archivo {nombre_archivo} no existe en el directorio '{directorio}'.")

# Unir todos los DataFrames en uno solo
if dfs:
    df_unido = pd.concat(dfs, ignore_index=True)

    # Asegurarse de que el DataFrame tiene las columnas 'open', 'close', 'high', 'low'
    if {'Open', 'Close', 'High', 'Low'}.issubset(df_unido.columns):

        # Filtrar los datos para obtener las últimas 2 semanas (14 días)
        df_2_semanas = df_unido.head(14)  # Los primeros 14 días (últimas 2 semanas)

        # Graficar los valores 'open', 'close', 'high', 'low'
        plt.figure(figsize=(10, 6))

        plt.plot(df_2_semanas['Open'].replace(',', '', regex=True).astype(float), color='blue', label='Open')
        plt.plot(df_2_semanas['Close'].replace(',', '', regex=True).astype(float), label='Close', color='green')
        plt.plot(df_2_semanas['High'].replace(',', '', regex=True).astype(float), label='High', color='red')
        plt.plot(df_2_semanas['Low'].replace(',', '', regex=True).astype(float), label='Low', color='orange')

        # Configuración del gráfico
        plt.title('Valores Open, Close, High, Low en 2 semanas')
        plt.xlabel('Días')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(14), [f'Día {i + 1}' for i in range(14)])

        # Mostrar el gráfico
        plt.show()

        # Salvaguardar el DataFrame concatenado en un archivo CSV
        nombre_archivo_unido = os.path.join(directorio, 'datos_unidos.csv')
        df_unido.to_csv(nombre_archivo_unido, index=False)
        print(f"El DataFrame concatenado se ha guardado en '{nombre_archivo_unido}'.")

    else:
        print("Faltan algunas columnas necesarias ('Open', 'Close', 'High', 'Low').")
else:
    print("No se encontraron archivos CSV para unir.")

