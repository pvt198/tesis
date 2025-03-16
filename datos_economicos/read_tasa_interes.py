import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV con encabezados y parsear fechas
df = pd.read_csv("tasa_interes.csv", parse_dates=["observation_date"])

# Renombrar columnas para mayor claridad
df.rename(columns={"observation_date": "Fecha", "DFF": "Tasa"}, inplace=True)

# Ordenar los datos por fecha
df = df.sort_values("Fecha")

# Resampleo trimestral: Tomar la última tasa de cada trimestre
df_trim = df.resample("Q", on="Fecha").last()

# Calcular la variación porcentual trimestre a trimestre
df_trim["Variacion_Trimestral"] = df_trim["Tasa"].pct_change() * 100

# Crear la gráfica de la tasa de interés
plt.figure(figsize=(10, 6))
plt.plot(df["Fecha"], df["Tasa"], linestyle="-", color="b", label="Tasa de Interés")
plt.xlabel("Fecha")
plt.ylabel("Tasa de Interés (%)")
plt.title("Evolución de la Tasa de Interés")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Crear la gráfica de la variación trimestral
plt.figure(figsize=(10, 6))
plt.plot(df_trim.index, df_trim["Variacion_Trimestral"], linestyle="-", color="r", marker="o", label="Variación Trimestral")
plt.xlabel("Fecha")
plt.ylabel("Variación (%)")
plt.title("Variación Trimestral de la Tasa de Interés")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)  # Línea en 0% para referencia
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
