import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Make sure the models folder exists
os.makedirs('models', exist_ok=True)

# ---------------------------
# Model 1: Sentiment Analysis
# ---------------------------

# Columns for sentiment.csv as per your info
cols_sentiment = ['label', 'id', 'date', 'query', 'user', 'text']

sentiment_df = pd.read_csv("sentiment.csv", header=None, names=cols_sentiment, encoding='latin1')
sentiment_df.dropna(subset=['text', 'label'], inplace=True)

X_sent = sentiment_df['text']
y_sent = sentiment_df['label']

X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
    X_sent, y_sent, test_size=0.2, random_state=42
)

sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

sentiment_pipeline.fit(X_train_sent, y_train_sent)
joblib.dump(sentiment_pipeline, 'models/sentiment_model.pkl')
print("✅ Saved sentiment_model.pkl in 'models' folder")

print(f"Sentiment model train accuracy: {sentiment_pipeline.score(X_train_sent, y_train_sent):.4f}")
print(f"Sentiment model test accuracy: {sentiment_pipeline.score(X_test_sent, y_test_sent):.4f}")

# ------------------------
# Model 2: Toxicity Detection
# ------------------------


try:
    toxicity_df = pd.read_csv("comments.csv", encoding='latin1')
    
    # Check columns
    print("Toxicity CSV columns:", toxicity_df.columns.tolist())

    if 'text' not in toxicity_df.columns or 'label' not in toxicity_df.columns:
        raise ValueError("Toxicity dataset must contain 'text' and 'label' columns.")

    toxicity_df.dropna(subset=['text', 'label'], inplace=True)

    X_tox = toxicity_df['text']
    y_tox = toxicity_df['label']

    X_train_tox, X_test_tox, y_train_tox, y_test_tox = train_test_split(
        X_tox, y_tox, test_size=0.2, random_state=42
    )

    toxicity_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    toxicity_pipeline.fit(X_train_tox, y_train_tox)
    joblib.dump(toxicity_pipeline, 'models/toxicity_model.pkl')
    print("✅ Saved toxicity_model.pkl in 'models' folder")

    print(f"Toxicity model train accuracy: {toxicity_pipeline.score(X_train_tox, y_train_tox):.4f}")
    print(f"Toxicity model test accuracy: {toxicity_pipeline.score(X_test_tox, y_test_tox):.4f}")

except FileNotFoundError:
    print("❌ File 'toxicity_comments.csv' not found. Please provide toxicity dataset with 'text' and 'label' columns.")
except ValueError as ve:
    print(f"❌ {ve}")