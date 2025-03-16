import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cambiar opciones para imprimir el DataFrame completo
pd.set_option('display.max_rows', None)  # Mostrar todas las filas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.width', None)  # Ajustar automáticamente el ancho
pd.set_option('display.max_colwidth', None)  # Mostrar el contenido de las columnas


# Cargar datos desde CSV de Nasdaq
def load_nasdaq_data():
    df_nasdaq = pd.read_csv("datos_unidos.csv", dayfirst=True)

    # Convertir la columna 'Date' a datetime
    df_nasdaq['Date'] = pd.to_datetime(df_nasdaq['Date'], format='%m/%d/%Y', errors='coerce')

    # Comprobar si hay fechas no convertidas
    if df_nasdaq['Date'].isna().any():
        print("Algunas fechas no se pudieron convertir:")
        print(df_nasdaq[df_nasdaq['Date'].isna()])

    # Eliminar las comillas de los valores y convertir a float
    for col in ["Open", "High", "Low", "Close"]:
        df_nasdaq[col] = df_nasdaq[col].str.replace(",", "").str.replace('"', '').astype(float)

    return df_nasdaq


# Cargar datos macroeconómicos
def load_macro_data():
    # Lista de archivos CSV con datos macroeconómicos
    macro_files = ["GDP.csv", "desempleo.csv", "tasa_interes.csv", "M2.csv",
                   "inflation.csv"]  # Agrega más archivos si es necesario

    # Diccionario para mapear nombres de columnas a nombres más legibles
    rename_dict = {
        "GDP": "GDP",
        "UNRATE": "Unemployment Rate",
        "DFF": "Interest Rate",
        "M2SL": "M2 Money Supply",
        "CORESTICKM159SFRBATL": "Inflation"
    }

    # Lista para almacenar los DataFrames
    dataframes = []

    for file in macro_files:
        df = pd.read_csv(file, parse_dates=["observation_date"])
        df.rename(columns={"observation_date": "Date"}, inplace=True)

        # Convertir la columna 'Date' a datetime, asegurando formato correcto
        if file == "tasa_interes.csv":
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%d-%m', errors='coerce')
        else:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

        # Renombrar columnas para mayor claridad
        df.rename(columns=rename_dict, inplace=True)

        # Comprobar si hay fechas no convertidas
        if df['Date'].isna().any():
            print(f"Algunas fechas en {file} no se pudieron convertir:")
            print(df[df['Date'].isna()])

        dataframes.append(df)

    # Combinar todos los DataFrames en uno solo usando 'Date' como clave
    df_macro = dataframes[0]
    for df in dataframes[1:]:
        df_macro = pd.merge(df_macro, df, on="Date", how="outer")

    # Rellenar los valores NaN hacia adelante para tener datos completos en cada fecha
    df_macro = df_macro.sort_values("Date").ffill()

    return df_macro



# Crear un nuevo DataFrame combinando los datos del Nasdaq con los indicadores económicos
def combine_nasdaq_with_macroeconomic_data(df_nasdaq, expanded_macro):

    # Combinar los datos del Nasdaq con los datos expandidos del macroeconómico
    combined_df = pd.merge(df_nasdaq, expanded_macro, on='Date', how='left')

    return combined_df


# Cargar los datos macroeconómicos y de Nasdaq
df_nasdaq = load_nasdaq_data()
df_macro = load_macro_data()

df_macro_expanded = df_macro.set_index('Date').resample('D').ffill().reset_index()
df_macro_expanded['Date'] = df_macro_expanded['Date'].dt.strftime('%Y-%m-%d')
df_macro_expanded['Date'] = pd.to_datetime(df_macro_expanded['Date'])
df_combined = combine_nasdaq_with_macroeconomic_data(df_nasdaq, df_macro_expanded)
def round_to_nearest_half(x):
    return round(x / 0.5) * 0.5
df_combined['Interest Rate'] = df_combined['Interest Rate'].apply(round_to_nearest_half)
df_combined['Interest Rate'] = df_combined['Interest Rate'].rolling(window=30).max()

# Create a new DataFrame to store gradients
df_gradients = df_combined.copy()

# Calculate gradients and normalize
for col in ['GDP', 'Unemployment Rate', 'Interest Rate', 'M2 Money Supply', 'Inflation']:
    # Calculate gradient as the difference between current and previous value
    df_gradients[col] = df_gradients[col].diff().fillna(0)  # First row starts at zero

    # Normalize the gradient using the previous cell's value
    previous_value = df_macro[col].shift(-1).fillna(1)  # Fill NaN with 1 to avoid division by zero
    df_gradients[col] = df_gradients[col] / previous_value  # Normalization

df_gradients.to_csv('NASDAQ_price_plus_macro.csv', index=False)

### PLOTTING ###

import matplotlib.dates as mdates

# Create the main figure
fig, ax1 = plt.subplots(figsize=(15, 8))

# Format x-axis to show only the year
ax1.xaxis.set_major_locator(mdates.YearLocator())  # Show only the year
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as year
ax1.set_xlim([df_combined["Date"].min(), df_combined["Date"].max()])
ax1.tick_params(axis='x', labelsize=10)  # Reduce font size for readability
ax1.grid(True)

# Plot Nasdaq Closing Price
ax1.plot(df_combined["Date"], df_combined["Close"], label="Nasdaq Closing Price", color="blue", linewidth=4)
ax1.set_ylabel("Nasdaq Close", fontsize=10, color="blue")

# Create second y-axis for GDP
ax2 = ax1.twinx()
ax2.plot(df_combined["Date"], df_combined["M2 Money Supply"], label="M2 Money Supply", color="red", linewidth=2)
ax2.set_ylabel("M2 Money Supply", fontsize=10, color="red")

# Create third y-axis for Unemployment Rate
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(df_combined["Date"], df_combined["Inflation"], label="Inflation", color="gray",  linewidth=2)
ax3.set_ylabel("Inflation (%)", fontsize=10, color="gray")

# Combine all legends into one
lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left", fontsize=10)

# Title
plt.title("Nasdaq Closing Price vs Macroeconomic Indicators (2010-Present)", fontsize=12)

# Show plot
plt.show()



import matplotlib.dates as mdates

#Filter dataset from 2010 onwards
df_gradients = df_gradients[df_gradients["Date"] >= "2010-01-01"]

# Create the main figure
fig, ax1 = plt.subplots(figsize=(15, 8))

# Format x-axis to show only the year
ax1.xaxis.set_major_locator(mdates.YearLocator())  # Show only the year
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as year
ax1.set_xlim([df_gradients["Date"].min(), df_gradients["Date"].max()])
ax1.tick_params(axis='x', labelsize=10)  # Reduce font size for readability
ax1.grid(True)

# Plot Nasdaq Closing Price
ax1.plot(df_gradients["Date"], df_gradients["Close"], label="Nasdaq Closing Price", color="blue", linewidth=4)
ax1.set_ylabel("Nasdaq Close", fontsize=10, color="blue")

# Create second y-axis for GDP
ax2 = ax1.twinx()
ax2.plot(df_gradients["Date"], df_gradients["M2 Money Supply"], label="M2 Money Supply", color="red", linewidth=2)
ax2.set_ylabel("M2 Money Supply", fontsize=10, color="red")

# Create third y-axis for Unemployment Rate
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(df_gradients["Date"], df_gradients["Inflation"], label="Inflation", color="gray",  linewidth=2)
ax3.set_ylabel("Inflation", fontsize=10, color="gray")

# Combine all legends into one
lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left", fontsize=10)

# Title
plt.title("Nasdaq Closing Price vs Macroeconomic Indicators (2010-Present)", fontsize=12)

plt.tight_layout()

# Show plot
plt.show()

