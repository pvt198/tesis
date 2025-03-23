import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Propia API key de Alpha Vantage
API_KEY = "V7CCRIFNCAMMHMX8"


# Función para obtener noticias dentro de un mes específico
def obtener_noticias_mes(anio, mes):
    # Calcular la fecha de inicio y fin del mes
    fecha_inicio = datetime(anio, mes, 1)
    if mes == 12:
        fecha_fin = datetime(anio + 1, 1, 1) - timedelta(minutes=1)
    else:
        fecha_fin = datetime(anio, mes + 1, 1) - timedelta(minutes=1)

    # Formatear las fechas en el formato requerido por la API (YYYYMMDDTHHMM)
    time_from = fecha_inicio.strftime("%Y%m%dT0000")
    time_to = fecha_fin.strftime("%Y%m%dT2359")

    # Construir la URL con los parámetros
    url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=technology&apikey={API_KEY}&time_from={time_from}&time_to={time_to}&limit=1000")

    # Realizar la solicitud a la API
    response = requests.get(url)
    data = response.json()
    return data


# Función para calcular estadísticas del sentimiento
def calcular_estadisticas_sentimiento(datos):
    if "feed" in datos and datos["feed"]:
        puntajes = [articulo["overall_sentiment_score"] for articulo in datos["feed"]]
        return {
            "promedio": sum(puntajes) / len(puntajes),
            "min": min(puntajes),
            "max": max(puntajes),
            "cantidad": len(puntajes)
        }
    return None  # Si no hay noticias


# Definir el rango de tiempo (enero 2024 - diciembre 2024)
anio = 2024

# Listas para graficar
meses = []
promedios = []
mins = []
maxs = []
cantidades = []

# Iterar mes a mes
for mes in range(1, 13):
    datos_mes = obtener_noticias_mes(anio, mes)
    estadisticas = calcular_estadisticas_sentimiento(datos_mes)

    if estadisticas is not None:
        meses.append(mes)
        promedios.append(estadisticas["promedio"])
        mins.append(estadisticas["min"])
        maxs.append(estadisticas["max"])
        cantidades.append(estadisticas["cantidad"])
        print(
            f"Mes {mes}/{anio}: Sentimiento promedio = {estadisticas['promedio']:.4f}, Min = {estadisticas['min']:.4f}, Max = {estadisticas['max']:.4f}, Noticias = {estadisticas['cantidad']}")
    else:
        print(f"Mes {mes}/{anio}: No se encontraron noticias.")

# Graficar resultados
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(meses, promedios, marker='o', label='Promedio Sentimiento', color='b')
ax1.fill_between(meses, mins, maxs, color='b', alpha=0.1, label='Rango Sentimiento')
ax2.bar(meses, cantidades, alpha=0.3, color='g', label='Cantidad de Noticias')

ax1.set_xlabel('Mes')
ax1.set_ylabel('Sentimiento', color='b')
ax2.set_ylabel('Número de Noticias', color='g')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_xticks(meses)
ax1.grid(True)
plt.title(f"Sentimiento del mercado tecnológico en {anio}")
plt.show()
