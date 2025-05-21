from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load models on startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    sentiment_model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    toxicity_model = joblib.load(os.path.join(MODEL_DIR, "toxicity_model.pkl"))
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    sentiment_model = None
    toxicity_model = None

# Label maps (update if needed based on your model classes)
sentiment_label_map = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive"
}

toxicity_label_map = {
    0: "non-toxic",
    1: "toxic"
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text or not sentiment_model or not toxicity_model:
        return jsonify({"error": "Invalid input or model not loaded"}), 400

    try:
        sentiment_pred = int(sentiment_model.predict([text])[0])
        toxicity_pred = int(toxicity_model.predict([text])[0])

        # Safe mapping fallback
        sentiment_label = sentiment_label_map.get(sentiment_pred, str(sentiment_pred))
        toxicity_label = toxicity_label_map.get(toxicity_pred, str(toxicity_pred))

        return jsonify({
            "text": text,
            "sentiment": sentiment_label,
            "toxicity": toxicity_label
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
