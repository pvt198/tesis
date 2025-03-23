import os
import csv
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

#  Cargar el modelo FinBERT
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Función para análisis de sentimientos
def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1)
    labels = model.config.id2label

    sentiment_idx = torch.argmax(probs, dim=1).item()
    sentiment_label = labels[sentiment_idx]
    sentiment_score = probs[0][sentiment_idx].item()

    return sentiment_label, sentiment_score


# Carga CSV files con formado "news_2024-mm-dd.csv"
folder_path = "./"  # Change if needed
# Buscar archivos con formato "news_2024-mm-dd.csv" y "news_2025-mm-dd.csv"
csv_files = glob.glob(os.path.join(folder_path, "news_2024-??-??.csv")) + \
            glob.glob(os.path.join(folder_path, "news_2025-??-??.csv"))

# Salidas
dates = []
avg_weighted_sentiments = []

aggregated_results = [["Date", "Average Original Score", "Average FinBERT Score", "Average Weighted Sentiment"]]

# recurres todos los files
for file_path in sorted(csv_files):
    print(file_path)
    file_name = os.path.basename(file_path)
    date_str = file_name.replace("news_", "").replace(".csv", "")

    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Skipping invalid file: {file_name}")
        continue

    original_scores = []
    finbert_scores = []
    weighted_sentiments = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)
        data_rows = rows[1:]  # salta la header

        for row in data_rows:
            try:
                headline, description, original_score = row[0], row[1], float(row[2])
                full_text = f"{headline}. {description}"
                sentiment_label, finbert_score = sentiment_analysis(full_text)

                # da el sentimiento
                sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
                sentiment_value = sentiment_mapping.get(sentiment_label, 0)

                weighted_sentiment = sentiment_value * finbert_score

                original_scores.append(original_score)
                finbert_scores.append(finbert_score)
                weighted_sentiments.append(weighted_sentiment)
            except ValueError:
                print(f"Skipping invalid row in {file_name}: {row}")

    # hache la media de dia
    avg_original_score = sum(original_scores) / len(original_scores) if original_scores else 0
    avg_finbert_score = sum(finbert_scores) / len(finbert_scores) if finbert_scores else 0
    avg_weighted_sentiment = sum(weighted_sentiments) / len(weighted_sentiments) if weighted_sentiments else 0
    print(avg_weighted_sentiment)
    # Store results
    aggregated_results.append([date_str, avg_original_score, avg_finbert_score, avg_weighted_sentiment])
    dates.append(date)
    avg_weighted_sentiments.append(avg_weighted_sentiment)

# output
output_csv = "aggregated_sentiment.csv"
with open(output_csv, mode="w", newline="", encoding="utf-8") as out_file:
    writer = csv.writer(out_file)
    writer.writerows(aggregated_results)

print(f"Aggregated sentiment data saved to {output_csv}")

# Plot 1
plt.figure(figsize=(10, 5))
plt.plot(dates, avg_weighted_sentiments, marker="o", linestyle="-", color="b", label="Weighted Sentiment")
plt.xlabel("Date")
plt.ylabel("Average Weighted Sentiment")
plt.title("Market Sentiment Over Time")
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(True)
plt.legend()
plt.show()

# Plot 2
# Calcular estadísticas
mean_sentiment = np.mean(avg_weighted_sentiments)
variance_sentiment = np.var(avg_weighted_sentiments)
std_dev_sentiment = np.std(avg_weighted_sentiments)
min_sentiment = np.min(avg_weighted_sentiments)
max_sentiment = np.max(avg_weighted_sentiments)
plt.figure(figsize=(12, 6))
plt.plot(dates, avg_weighted_sentiments, marker="o", linestyle="-", color="b", label="Avg Weighted Sentiment")
plt.axhline(mean_sentiment, color="g", linestyle="--", label=f"Mean ({mean_sentiment:.4f})")
plt.fill_between(dates, mean_sentiment - std_dev_sentiment, mean_sentiment + std_dev_sentiment,
                 color="g", alpha=0.2, label="±1 Std Dev")
plt.fill_between(dates, min_sentiment, max_sentiment, color="b", alpha=0.1, label="Min-Max Range")
plt.xlabel("Date")
plt.ylabel("Sentiment Score")
plt.title("Market Sentiment Over Time")
plt.xticks(rotation=90)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
print(f"Mean Sentiment: {mean_sentiment:.4f}")
print(f"Variance: {variance_sentiment:.4f}")
print(f"Standard Deviation: {std_dev_sentiment:.4f}")
print(f"Min Sentiment: {min_sentiment:.4f}")
print(f"Max Sentiment: {max_sentiment:.4f}")