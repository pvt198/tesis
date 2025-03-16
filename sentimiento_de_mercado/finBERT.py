from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# Cargar el modelo FinBERT
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Función para análisis de sentimientos
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Obtener logits del modelo
    with torch.no_grad():
        logits = model(**inputs).logits

    # Aplicar softmax para convertir logits en probabilidades
    probs = softmax(logits, dim=1)

    # Etiquetas correctas del modelo
    labels = model.config.id2label  # Esto obtiene las etiquetas reales del modelo

    # Obtener el índice con la mayor probabilidad
    sentiment_idx = torch.argmax(probs, dim=1).item()
    sentiment_label = labels[sentiment_idx]  # Asignar la etiqueta correcta

    return sentiment_label, probs[0][sentiment_idx].item(), logits

# Texto de prueba
text = "The company's financial performance has exceeded expectations, and its stock price has risen sharply."

# Obtener sentimiento
sentiment, score, logits = sentiment_analysis(text)

# Mostrar resultados
print(f"Sentiment: {sentiment} (Score: {score:.4f})")
