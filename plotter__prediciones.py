import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

files = ["./results/OPRICE_RNN_Precio_y_macro_predictions.csv","./results/RNN_Precio_y_macro_predictions.csv",]
labels = ["RNN Precio",
          "RNN Precio y Macro",
         ]
plt.figure(figsize=(12, 6))

start = 0
stop = -1
for label, file in zip(labels, files):
    df = pd.read_csv(file)
    pred_dates = pd.to_datetime(df['Date'])
    print(pred_dates)

    if label == "RNN Precio":
        plt.plot(pred_dates[start:stop],df['Actual_Close'][start:stop], label=f'Actual Price', color="black")
        plt.plot(pred_dates[start:stop],df['Predicted_Close'][start:stop], label=f'Predicted {label}')
    else:
        plt.plot(pred_dates[start:stop],df['Predicted_Close'][start:stop], label=f'Predicted {label}')
    #plt.gca().xaxis.set_major_locator(mdates.YearLocator())

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.title('Comparación de Precios de Cierre Reales vs Predichos')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()