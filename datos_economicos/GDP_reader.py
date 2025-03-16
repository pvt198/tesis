import pandas as pd
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos desde el archivo CSV
data = pd.read_csv("GDP.csv", parse_dates=["observation_date"])

# Paso 2: Graficar el PIB a lo largo del tiempo
plt.figure(figsize=(10, 6))
plt.plot(data['observation_date'], data['GDP'], label='PIB', color='blue')
plt.title('PIB a lo largo del tiempo')
plt.xlabel('Fecha')
plt.ylabel('PIB')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Paso 3: Calcular la variación porcentual del PIB trimestre a trimestre
data['GDP_pct_change'] = data['GDP'].pct_change() * 100

# Paso 4: Graficar la variación porcentual a lo largo del tiempo
plt.figure(figsize=(10, 6))
plt.plot(data['observation_date'], data['GDP_pct_change'], label='Cambio porcentual trimestral', color='red')
plt.title('Cambio porcentual trimestral del PIB')
plt.xlabel('Fecha')
plt.ylabel('Cambio porcentual')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
