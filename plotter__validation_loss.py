import pandas as pd
import matplotlib.pyplot as plt

files = ["./results/RNN_solo_Precio_val_loss_history.csv",
         "./results/RNN_Precio_y_macro_val_loss_history.csv",]
labels = ["RNN Solo Precio",
         "RNN Precio y Macro",
         ]


plt.figure(figsize=(10, 5))
for label, file in zip(labels, files):
    df = pd.read_csv(file)
    plt.plot(df["Epoch"][:50], df["Val_Loss"][:50], label=label)

plt.xlabel("Época")
plt.ylabel("Pérdida de Validación (Val_Loss)")
plt.title("Comparación de Pérdida de Validación")
plt.legend()
plt.grid()
plt.show()
