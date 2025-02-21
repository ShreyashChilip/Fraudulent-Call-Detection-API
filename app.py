import os
import joblib
import re
import string
import assemblyai as aai
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
aai.settings.api_key = "afc345dcecac49d3b711cd1a9b59757e"

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

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    """API endpoint to predict scam probability from an audio file."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        file_path = "temp_audio.wav"
        audio_file.save(file_path)

        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = aai.Transcriber().transcribe(file_path, config)

        speaker_texts = {}
        for utterance in transcript.utterances:
            speaker = f"Speaker {utterance.speaker}"
            text = utterance.text.strip()
            speaker_texts.setdefault(speaker, []).append(text)

        speaker_text_strings = {speaker: ". ".join(texts) + "." for speaker, texts in speaker_texts.items()}

        scam_confidences = []
        results = {}
        for speaker, text in speaker_text_strings.items():
            processed_text = preprocess_text(text)
            if not processed_text:
                results[speaker] = {"prediction": "NOT SCAM", "confidence": 0.0, "message": "No valid text"}
            else:
                text_vectorized = vectorizer.transform([processed_text])
                probability = model.predict_proba(text_vectorized)[0][1]
                prediction = "SCAM" if probability >= 0.60 else "NOT SCAM"
                message = get_warning_message(probability)
                results[speaker] = {"prediction": prediction, "confidence": round(probability, 2), "message": message}
                if prediction == "SCAM":
                    scam_confidences.append(probability)

        final_confidence = round(sum(scam_confidences) / len(scam_confidences), 2) if scam_confidences else 0.0
        final_prediction = "SCAM" if scam_confidences else "NOT SCAM"

        print(speaker_text_strings)

        return jsonify({
            "final_prediction": final_prediction,
            "final_confidence": final_confidence,
            "speaker_details": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
