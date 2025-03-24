import pandas as pd
import matplotlib.pyplot as plt

files = ["./results/LSTM_Precio_y_macro_OPRICE_val_loss_history.csv",
         "./results/LSTM_Precio_y_macro_val_loss_history.csv",]

labels = ["LSTM Only Price",
          "LSTM Price and MacroIndicators",

         ]



plt.figure(figsize=(10, 5))
for label, file in zip(labels, files):
    df = pd.read_csv(file)
    plt.plot(df["Epoch"], df["Val_Loss"], label=label)

plt.xlabel("Época")
plt.ylabel("Pérdida de Validación (Val_Loss)")
plt.title("Comparación de Pérdida de Validación")
plt.legend()
plt.grid()
plt.show()
