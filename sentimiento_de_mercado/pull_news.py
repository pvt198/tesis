import requests
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime, timedelta
import random

# API Key
API_KEY = "FAQE6GTDR1LHTPB2"

user_agents = [
    # Windows (Chrome, Edge, Firefox)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/119.0.0.0 Safari/537.36",

    # MacOS (Safari, Chrome, Firefox)
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.1 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_4) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/114.0",

    # Linux (Chrome, Firefox)
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:105.0) Gecko/20100101 Firefox/105.0",

    # Mobile (Android, iOS)
    "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 15_6 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Version/15.6 Mobile/15E148 Safari/537.36"
]


def obtener_noticias_dia(anio, mes, dia):
    fecha = datetime(anio, mes, dia)
    time_from = fecha.strftime("%Y%m%dT0000")
    time_to = fecha.strftime("%Y%m%dT2359")
    url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&sort=RELEVANCE&topics=technology"
           f"&apikey={API_KEY}&time_from={time_from}&time_to={time_to}&limit=1000")
    headers = {"User-Agent": random.choice(user_agents)}
    response = requests.get(url, headers=headers)
    data = response.json()

    # Save raw data to CSV per day
    filename = f"news_{anio}-{mes:02d}-{dia:02d}.csv"
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Summary", "Sentiment Score"])
        if "feed" in data:
            for article in data["feed"]:
                writer.writerow(
                    [article.get("title", ""), article.get("summary", ""), article.get("overall_sentiment_score", "")])

    print(f"Saved data to {filename}")
    return data


def calcular_estadisticas_sentimiento(datos):
    if "feed" in datos and datos["feed"]:
        puntajes = [articulo["overall_sentiment_score"] for articulo in datos["feed"]]
        return {
            "promedio": np.mean(puntajes),
            "desviacion": np.std(puntajes),
            "cantidad": len(puntajes)
        }
    return None


anio, mes = 2025, 3
num_dias = 31
num_llamadas = 25

# Espaciamos las llamadas uniformemente
dias_seleccionados = np.linspace(1, num_dias, num_llamadas, dtype=int)

fechas, promedios, desviaciones, cantidades = [], [], [], []

with open("sentimiento_tecnologia_marc2025.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Fecha", "Promedio Sentimiento", "Desviación", "Cantidad de Noticias"])

    for dia in dias_seleccionados:
        datos_dia = obtener_noticias_dia(anio, mes, dia)
        estadisticas = calcular_estadisticas_sentimiento(datos_dia)

        if estadisticas is not None:
            fechas.append(datetime(anio, mes, dia))
            promedios.append(estadisticas["promedio"])
            desviaciones.append(estadisticas["desviacion"])
            cantidades.append(estadisticas["cantidad"])
            writer.writerow([fechas[-1].strftime('%Y-%m-%d'), promedios[-1], desviaciones[-1], cantidades[-1]])

# Interpolación para completar los días faltantes
dias_totales = np.arange(1, num_dias + 1)
promedios_interp = np.interp(dias_totales, dias_seleccionados, promedios)
desviaciones_interp = np.interp(dias_totales, dias_seleccionados, desviaciones)

