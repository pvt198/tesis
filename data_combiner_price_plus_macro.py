import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from scipy.signal import savgol_filter


def load_nasdaq_data():
    df_nasdaq = pd.read_csv("datos_unidos.csv", dayfirst=True)
    df_nasdaq['Date'] = pd.to_datetime(df_nasdaq['Date'], format='%m/%d/%Y', errors='coerce')
    if df_nasdaq['Date'].isna().any():
        print("Algunas fechas no se pudieron convertir:")
        print(df_nasdaq[df_nasdaq['Date'].isna()])
    for col in ["Open", "High", "Low", "Close"]:
        df_nasdaq[col] = df_nasdaq[col].str.replace(",", "").str.replace('"', '').astype(float)
    return df_nasdaq

def load_macro_data():
    macro_files = ["GDP.csv", "desempleo.csv", "tasa_interes.csv", "M2.csv",
                   "inflation.csv"]  # Agrega más archivos si es necesario
    rename_dict = {
        "GDP": "GDP",
        "UNRATE": "Unemployment Rate",
        "DFF": "Interest Rate",
        "M2SL": "M2 Money Supply",
        "CORESTICKM159SFRBATL": "Inflation"
    }
    dataframes = []
    for file in macro_files:
        df = pd.read_csv("./datos_economicos/"+file, parse_dates=["observation_date"])
        df.rename(columns={"observation_date": "Date"}, inplace=True)
        if file == "tasa_interes.csv":
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        else:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        df.rename(columns=rename_dict, inplace=True)
        if df['Date'].isna().any():
            print(f"Algunas fechas en {file} no se pudieron convertir:")
            print(df[df['Date'].isna()])
        dataframes.append(df)

    # Combinar todos los DataFrames en uno solo usando 'Date' como clave
    df_macro = dataframes[0]
    for df in dataframes[1:]:
        df_macro = pd.merge(df_macro, df, on="Date", how="outer")
    df_macro = df_macro.sort_values("Date").ffill()
    return df_macro


def combine_nasdaq_with_macroeconomic_data(df_nasdaq, expanded_macro):
    combined_df = pd.merge(df_nasdaq, expanded_macro, on='Date', how='left')
    return combined_df


# Cargar los datos
df_nasdaq = load_nasdaq_data()
df_macro = load_macro_data()
df_macro_expanded = df_macro.set_index('Date').resample('D').ffill().reset_index()
df_macro_expanded['Date'] = df_macro_expanded['Date'].dt.strftime('%Y-%m-%d')
df_macro_expanded['Date'] = pd.to_datetime(df_macro_expanded['Date'])

df_combined = combine_nasdaq_with_macroeconomic_data(df_nasdaq, df_macro_expanded)
print(df_combined.iloc[0:1001])

def round_to_nearest_half(x):
    return round(x / 0.5) * 0.5
#df_combined['Interest Rate'] = savgol_filter(df_combined['Interest Rate'], window_length=31, polyorder=2)
df_combined['Interest Rate'] = df_combined['Interest Rate'].fillna(method='bfill')
df_combined['Interest Rate'] = df_combined['Interest Rate'].rolling(window=90).mean()
df_combined['Interest Rate'] = df_combined['Interest Rate'].fillna(method='bfill')
df_combined['Interest Rate'] = df_combined['Interest Rate'].apply(round_to_nearest_half)

df_combined.to_csv('./results/NASDAQ_price_plus_macro.csv', index=False)

# Nuevo dataframe para los gradientes
df_gradients = df_combined.copy()
# Calculo y normalizazion de los gradientes
for col in ['GDP', 'Unemployment Rate', 'Interest Rate', 'M2 Money Supply', 'Inflation']:
    df_gradients[col] = df_gradients[col].diff().fillna(0)
    previous_value = df_macro[col].shift(-1).fillna(1)
    df_gradients[col] = df_gradients[col] / previous_value  # Normalization
    if col == "Interest Rate":
        df_gradients[col] = df_gradients[col].apply(lambda x: x if -1 <= x <= 1 else None)
        df_gradients[col].fillna(method='ffill', inplace=True)
    else:
        df_gradients[col] = np.clip(df_gradients[col], -1, 1)

#df_gradients.to_csv('./results/NASDAQ_price_plus_macro.csv', index=False)
print(df_gradients.head())

dato1 = "Interest Rate"
dato2 = "Inflation"

### PLOT1 ###
fig, ax1 = plt.subplots(figsize=(15, 8))
ax1.xaxis.set_major_locator(mdates.YearLocator())  # Show only the year
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as year
ax1.set_xlim([df_combined["Date"].min(), df_combined["Date"].max()])
ax1.tick_params(axis='x', labelsize=10)  # Reduce font size for readability
ax1.grid(True)
ax1.plot(df_combined["Date"], df_combined["Close"], label="Nasdaq Closing Price", color="blue", linewidth=4)
ax1.set_ylabel("Nasdaq Close", fontsize=10, color="blue")
plt.xticks(rotation=90)
ax2 = ax1.twinx()
ax2.plot(df_combined["Date"], df_combined[dato1], label=dato1, color="red", linewidth=2)
ax2.set_ylabel(dato1, fontsize=10, color="red")
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(df_combined["Date"], df_combined[dato2], label=dato2, color="gray",  linewidth=2)
ax3.set_ylabel(dato2, fontsize=10, color="gray")
lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left", fontsize=10)
plt.title("Nasdaq Closing Price vs Macroeconomic Indicators (1990-Present)", fontsize=12)
plt.show()

### PLOT2 ###
fig, ax1 = plt.subplots(figsize=(15, 8))
ax1.xaxis.set_major_locator(mdates.YearLocator())  # Show only the year
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as year
ax1.set_xlim([df_gradients["Date"].min(), df_gradients["Date"].max()])
ax1.tick_params(axis='x', labelsize=10)  # Reduce font size for readability
ax1.grid(True)
ax1.plot(df_gradients["Date"], df_gradients["Close"], label="Nasdaq Closing Price", color="blue", linewidth=4)
ax1.set_ylabel("Nasdaq Close", fontsize=10, color="blue")
plt.xticks(rotation=90)
ax2 = ax1.twinx()
ax2.plot(df_gradients["Date"], df_gradients[dato1], label=dato1, color="red", linewidth=2)
ax2.set_ylabel(dato1, fontsize=10, color="red")
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(df_gradients["Date"], df_gradients[dato2], label=dato2, color="gray",  linewidth=2)
ax3.set_ylabel(dato2, fontsize=10, color="gray")
lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left", fontsize=10)
plt.title("Nasdaq Closing Price vs Macroeconomic Indicators (1990-Present)", fontsize=12)
plt.tight_layout()

plt.show()

