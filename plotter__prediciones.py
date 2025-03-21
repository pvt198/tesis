import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

files = ["./results/RNN_solo_Precio_predictions.csv",
         "./results/RNN_Precio_y_macro_predictions.csv",]
labels = ["RNN Solo Precio",
         "RNN Precio y Macro",
         ]

plt.figure(figsize=(12, 6))

for label, file in zip(labels, files):
    df = pd.read_csv(file)
    pred_dates = pd.to_datetime(df['Date'])
    print(pred_dates)
    plt.xticks(rotation=90)
    if label == "RNN Solo Precio":
        plt.plot(pred_dates, df['Actual_Close'], label=f'Actual {label}', color="black")
    plt.plot(pred_dates, df['Predicted_Close'], label=f'Predicted {label}')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.title('Comparación de Precios de Cierre Reales vs Predichos')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()