from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# Cargar el modelo FinBERT y su tokenizador
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Función para el análisis de sentimiento
def sentiment_analysis(text):
    # Tokenizar el texto de entrada y convertirlo en tensores de PyTorch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Obtener los logits (valores sin procesar) del modelo sin calcular gradientes
    with torch.no_grad():
        logits = model(**inputs).logits

    # Aplicar la función softmax para convertir los logits en probabilidades
    probs = softmax(logits, dim=1)

    # Extraer las probabilidades de cada clase de sentimiento
    prob_positive = probs[0][0].item()  # Probabilidad de ser positivo
    prob_neutral = probs[0][1].item()   # Probabilidad de ser neutral
    prob_negative = probs[0][2].item()  # Probabilidad de ser negativo

    # Obtener la etiqueta de sentimiento con la mayor probabilidad
    sentiment_idx = torch.argmax(probs, dim=1).item()  # Índice del sentimiento más probable
    sentiment_label = model.config.id2label[sentiment_idx]  # Obtener la etiqueta correspondiente

    return sentiment_label, prob_positive, prob_neutral, prob_negative

# Texto de ejemplo para analizar
text = "El desempeño financiero de la empresa ha superado las expectativas y el precio de sus acciones ha aumentado considerablemente."

# Obtener el análisis de sentimiento
sentiment, prob_positive, prob_neutral, prob_negative = sentiment_analysis(text)

# Mostrar los resultados
print(f"Sentimiento: {sentiment}")
print(f"Probabilidad - Positivo: {prob_positive:.4f}, Neutral: {prob_neutral:.4f}, Negativo: {prob_negative:.4f}")
