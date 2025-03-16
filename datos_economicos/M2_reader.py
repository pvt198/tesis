import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV con encabezados y parsear fechas
df = pd.read_csv("M2.csv", parse_dates=["observation_date"])

# Renombrar columnas para mayor claridad
df.rename(columns={"observation_date": "Fecha", "M2SL": "M2 (Oferta Monetaria)"}, inplace=True)

# Ordenar los datos por fecha
df = df.sort_values("Fecha")

# Calcular la variación porcentual trimestral
df["Variación Trimestral (%)"] = df["M2 (Oferta Monetaria)"].pct_change(periods=3) * 100

# ---------------- FIGURA 1: Evolución de M2 ----------------
plt.figure(figsize=(10, 5))
plt.plot(df["Fecha"], df["M2 (Oferta Monetaria)"], color="b", label="M2 (Oferta Monetaria)")
plt.xlabel("Fecha")
plt.ylabel("M2 (Miles de Millones de USD)")
plt.title("Evolución de la Oferta Monetaria M2 en EE.UU.")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()  # Mostrar la primera figura

# ---------------- FIGURA 2: Variación Trimestral de M2 ----------------
plt.figure(figsize=(10, 6))
plt.plot(df["Fecha"], df["Variación Trimestral (%)"], color="r", label="Variación Trimestral (%)")
plt.axhline(0, color="gray", linestyle="dotted")  # Línea de referencia en 0%
plt.xlabel("Fecha")
plt.ylabel("Variación Trimestral (%)")
plt.title("Variación Porcentual Trimestral de M2 en EE.UU.")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()  # Mostrar la segunda figura
