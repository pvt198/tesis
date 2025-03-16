import pandas as pd
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos desde el archivo CSV
data = pd.read_csv("Inflation.csv", parse_dates=["observation_date"])

# Renombrar la columna de inflación para mayor claridad
data.rename(columns={"CORESTICKM159SFRBATL": "Inflation"}, inplace=True)

# Paso 2: Calcular la variación trimestral porcentual
data["Quarterly_Change"] = data["Inflation"].pct_change(periods=3) * 100

# Paso 3: Graficar la inflación a lo largo del tiempo
plt.figure(figsize=(10, 6))
plt.plot(data['observation_date'], data['Inflation'], label='Inflación', color='red')
plt.title('Tasa de Inflación a lo Largo del Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Inflación (%)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Paso 4: Graficar la variación trimestral a lo largo del tiempo
plt.figure(figsize=(10, 6))
plt.plot(data['observation_date'], data['Quarterly_Change'], label='Variación Trimestral (%)', color='blue')
plt.title('Variación Trimestral de la Inflación')
plt.xlabel('Fecha')
plt.ylabel('Variación Trimestral (%)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
