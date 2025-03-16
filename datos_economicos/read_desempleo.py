import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV con encabezados y parsear fechas
df = pd.read_csv("desempleo.csv", parse_dates=["observation_date"])

# Renombrar columnas para mayor claridad
df.rename(columns={"observation_date": "Fecha", "UNRATE": "Tasa de Desempleo"}, inplace=True)

# Ordenar los datos por fecha (por si acaso)
df = df.sort_values("Fecha")

# Calcular la tasa de variación trimestral
df["Variacion Trimestral"] = df["Tasa de Desempleo"].pct_change(periods=3) * 100

# Crear la gráfica de la tasa de desempleo
plt.figure(figsize=(10, 5))
plt.plot(df["Fecha"], df["Tasa de Desempleo"], linestyle="-", color="b", label="Tasa de Desempleo")

# Personalizar el gráfico
plt.xlabel("Fecha")
plt.ylabel("Tasa de Desempleo (%)")
plt.title("Evolución de la Tasa de Desempleo en EE.UU.")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Mostrar la gráfica
plt.show()

# Crear la gráfica de la variación trimestral
plt.figure(figsize=(10, 5))
plt.plot(df["Fecha"], df["Variacion Trimestral"], color="r", label="Variación Trimestral (%)")

# Personalizar el gráfico
plt.xlabel("Fecha")
plt.ylabel("Variación Trimestral (%)")
plt.ylim([-20,20])
plt.title("Variación Trimestral de la Tasa de Desempleo en EE.UU.")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Mostrar la gráfica
plt.show()
