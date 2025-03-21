import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV con las predicciones
df = pd.read_csv("./results/RNN_Precio_y_macro_predictions.csv")

# Asegurar que la columna "Date" sea de tipo datetime y ordenar por fecha
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Inicialización de variables para el capital y la estrategia de trading
capital_inicial = 10000  # Capital inicial en dólares
capital = capital_inicial
acciones = 0
ganancia = 0
registro_transacciones = []

# Listas para almacenar los puntos de compra y venta
buy_dates, buy_prices, buy_predicted_prices = [], [], []
sell_dates, sell_prices, sell_predicted_prices = [], [], []

# Inicializar lista para almacenar el capital a lo largo del tiempo
capital_history = []

# Contadores para trades ganadores y perdedores
trades_ganadores = 0
trades_perdedores = 0
capital_antes_compra = 0
capital_antes_ventas = 0

# Simulación de trading
for i in range(len(df)-1):  # Iterar sobre todos los días

    precio_actual = df.loc[i, "Actual_Close"]  # Precio de cierre actual
    precio_predicho_siguiente = df.loc[i + 1, "Predicted_Close"]  # Precio predicho para el siguiente día

    # Se compra si se espera que el precio suba y hay capital disponible
    if  precio_predicho_siguiente > precio_actual and capital > 0:
        # Guardar el capital antes de comprar
        capital_antes_compra = capital
        acciones = capital / precio_actual  # Comprar tantas acciones como sea posible
        capital = 0  # No queda capital
        registro_transacciones.append((df.loc[i, "Date"], "BUY", precio_actual, acciones))
        buy_dates.append(df.loc[i, "Date"])
        buy_prices.append(precio_actual)
        buy_predicted_prices.append(precio_predicho_siguiente)  # Precio predicho al día siguiente


    # Se vende si se espera que el precio baje y tenemos acciones
    elif precio_predicho_siguiente < precio_actual and acciones > 0:
        # Guardar el capital antes de vender
        capital_antes_venta = acciones * precio_actual  # Capital al vender las acciones
        capital = capital_antes_venta  # Vender todas las acciones
        acciones = 0  # No quedan acciones en posesión
        registro_transacciones.append((df.loc[i, "Date"], "SELL", precio_actual, capital))
        sell_dates.append(df.loc[i, "Date"])
        sell_prices.append(precio_actual)
        sell_predicted_prices.append(precio_predicho_siguiente)  # Precio predicho al día siguiente

        # Contar el trade como ganador o perdedor comparando el capital
        if capital_antes_venta > capital_antes_compra:  # Trade ganador
            trades_ganadores += 1
        else:  # Trade perdedor
            trades_perdedores += 1


    # Guardar el capital actual en la historia
    capital_history.append(capital + acciones * precio_actual)

# Si al final del período aún tenemos acciones, venderlas al último precio disponible
if acciones > 0:
    precio_final = df.loc[len(df) - 1, "Actual_Close"]
    capital = acciones * precio_final
    ganancia = capital - capital_inicial
    registro_transacciones.append((df.loc[len(df) - 1, "Date"], "SELL (Final)", precio_final, capital))
    sell_dates.append(df.loc[len(df) - 1, "Date"])
    sell_prices.append(precio_final)
    sell_predicted_prices.append(precio_final)  # Último precio no tiene predicción siguiente

# Guardar las transacciones en un archivo CSV
df_transacciones = pd.DataFrame(registro_transacciones, columns=["Fecha", "Acción", "Precio", "Capital/Shares"])
df_transacciones.to_csv("./results/resultados_trading.csv", index=False)

# Calcular el porcentaje de trades ganadores
total_trades = trades_ganadores + trades_perdedores
porcentaje_ganadores = (trades_ganadores / total_trades * 100) if total_trades > 0 else 0

# Imprimir resultados
print(f"Total de trades: {total_trades}")
print(f"Trades ganadores: {trades_ganadores}")
print(f"Trades perdedores: {trades_perdedores}")
print(f"Porcentaje de trades ganadores: {porcentaje_ganadores:.2f}%")

# GRAFICO SERIE TEMPORAL
plt.figure(figsize=(14, 7))
plt.plot(df["Date"], df["Actual_Close"], label="Precio Real", color="black", linewidth=2)
plt.plot(df["Date"], df["Predicted_Close"], label="Precio Predicho", color="gray", linestyle="dashed", linewidth=1)
plt.scatter(buy_dates, buy_prices, marker="^", color="blue", label="Compra (BUY)", s=100)
plt.scatter(sell_dates, sell_prices, marker="v", color="red", label="Venta (SELL)", s=100)
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.title("Precio Real vs. Predicho con Puntos de Compra/Venta")
plt.legend()
plt.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.xticks(rotation=90)
plt.show()

# GRAFICO ZOOM EN UN MES
mes_zoom = "2022-01"  # mes de observacion
df_zoom = df[df["Date"].dt.strftime("%Y-%m") == mes_zoom]
buy_zoom_dates = [d for d in buy_dates if d in df_zoom["Date"].values]
buy_zoom_prices = [p for d, p in zip(buy_dates, buy_prices) if d in df_zoom["Date"].values]
buy_zoom_predicted = [p for d, p in zip(buy_dates, buy_predicted_prices) if d in df_zoom["Date"].values]
sell_zoom_dates = [d for d in sell_dates if d in df_zoom["Date"].values]
sell_zoom_prices = [p for d, p in zip(sell_dates, sell_prices) if d in df_zoom["Date"].values]
sell_zoom_predicted = [p for d, p in zip(sell_dates, sell_predicted_prices) if d in df_zoom["Date"].values]
plt.figure(figsize=(14, 7))
plt.plot(df_zoom["Date"], df_zoom["Actual_Close"], label="Precio Real", color="black", linewidth=2)
plt.plot(df_zoom["Date"], df_zoom["Predicted_Close"], label="Precio Predicho", color="gray", linewidth=1, linestyle="dashed")
plt.scatter(buy_zoom_dates, buy_zoom_prices, marker="^", color="blue", label="Compra (BUY)", s=100)
plt.scatter(sell_zoom_dates, sell_zoom_prices, marker="v", color="red", label="Venta (SELL)", s=100)
plt.scatter(buy_zoom_dates, buy_zoom_predicted, marker="o", color="blue", alpha=0.5, s=80, label="Precio Predicho Día Siguiente (BUY)")
plt.scatter(sell_zoom_dates, sell_zoom_predicted, marker="o", color="red", alpha=0.5, s=80, label="Precio Predicho Día Siguiente (SELL)")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.title(f"Zoom en {mes_zoom}: Precio Real vs. Predicho con Puntos de Compra/Venta")
plt.legend()
plt.grid()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# GRAFICO VARIACIÓN DEL CAPITAL A LO LARGO DEL TIEMPO
precio_inicial = df.loc[0, "Actual_Close"]
acciones_iniciales = capital_inicial / precio_inicial
capital_buy_and_hold = acciones_iniciales * df["Actual_Close"]
# Asegurarse de que ambas listas tengan la misma longitud
while len(capital_history) < len(df):
    capital_history.append(capital_history[-1])  # Rellenar con el último capital
plt.figure(figsize=(14, 7))
plt.plot(df["Date"], capital_history, label="Capital Estrategia Activa", color="blue", linewidth=2)
plt.plot(df["Date"], capital_buy_and_hold, label="Capital Buy and Hold", color="orange", linestyle="dashed", linewidth=2)
plt.xlabel("Fecha")
plt.ylabel("Capital ($)")
plt.title("Evolución del Capital: Estrategia Activa vs. Buy and Hold")
plt.legend()
plt.grid()
plt.xticks(rotation=90)
plt.show()

# GRAFICO BOX PLOT DEL CAPITAL
plt.figure(figsize=(14, 7))
box_data = [capital_history, capital_buy_and_hold]
box_colors = ['blue', 'orange']
bp = plt.boxplot(box_data, vert=False, patch_artist=True)
for i in range(len(bp['boxes'])):
    bp['boxes'][i].set_facecolor(box_colors[i])
    bp['boxes'][i].set_linewidth(1)
    bp['boxes'][i].set_edgecolor('black')
plt.title("Box Plot del Capital: Estrategia Activa vs. Buy and Hold")
plt.xlabel("Capital ($)")
plt.yticks([1, 2], ["Estrategia Activa", "Buy and Hold"])
plt.grid()
plt.show()
