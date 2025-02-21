from flask import Flask, request, jsonify
import joblib
import re
import string

app = Flask(__name__)

# Load ML model and vectorizer
model = joblib.load("models/logistic_regression_model_new.pkl")
vectorizer = joblib.load("models/vectorizer_new.pkl")

def preprocess_text(text):
    """Preprocess text consistently with training data."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_warning_message(probability):
    """Generate a warning message based on confidence level."""
    if probability >= 0.90:
        return "☠️ This call is highly suspicious! Disconnect immediately."
    elif probability >= 0.75:
        return "🚨 Strong scam indicators detected! Be cautious."
    elif probability >= 0.60:
        return "⚠️ Some scam-like patterns detected."
    elif probability >= 0.50:
        return "🚧 Slightly below scam threshold, verify details."
    else:
        return "✅ No scam detected, but stay cautious."

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict scam probability."""
    try:
        data = request.json
        text = data.get("text", "")

        processed_text = preprocess_text(text)
        if not processed_text:
            return jsonify({"error": "Invalid or empty text"}), 400

        # Vectorize and predict
        text_vectorized = vectorizer.transform([processed_text])
        probability = model.predict_proba(text_vectorized)[0][1]
        prediction = "SCAM" if probability >= 0.6 else "NOT SCAM"
        message = get_warning_message(probability)

        return jsonify({
            "prediction": prediction,
            "confidence": round(probability, 2),
            "message": message
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)