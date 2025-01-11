from flask import Flask, request, jsonify
from flask_cors import CORS  
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle  

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


try:
    model = tf.keras.models.load_model("hate_speech_model.h5")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please check the path.")
    model = None


try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print("Tokenizer file not found. Please check the path.")
    tokenizer = None


def clean_text(text):
    """Clean and preprocess the text."""
    text = text.lower()  # Lowercase the text
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.strip()  # Remove extra whitespace
    return text


@app.route("/predict_batch", methods=["POST", "OPTIONS"])
def predict_batch():
    # Handle OPTIONS preflight requests
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight successful"}), 200

    if not model or not tokenizer:
        return jsonify({"error": "Model or tokenizer not loaded"}), 500

    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "No texts provided"}), 400

    texts = data["texts"]
    results = []

    for entry in texts:
        if "username" not in entry or "text" not in entry:
            return jsonify({"error": "Each entry must contain 'username' and 'text'"}), 400

        username = entry["username"]
        raw_text = entry["text"]

        # Clean and preprocess the text
        cleaned_text = clean_text(raw_text)

        # Tokenize and pad the input text
        examples_cleaned = [cleaned_text]
        examples_seq = tokenizer.texts_to_sequences(examples_cleaned)
        examples_padded = pad_sequences(examples_seq, maxlen=100, padding='post')

        # Predict with the model
        predictions = model.predict(examples_padded)
        predicted_label = predictions.argmax(axis=1)[0]

        # Map the predicted label
        if predicted_label == 0:
            predicted_category = "hate_speech"
            is_filtered = True
        elif predicted_label == 1:
            predicted_category = "neither"
            is_filtered = False

        results.append({
            "username": username,
            "text": raw_text,
            "category": predicted_category,
            "filtered": is_filtered
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
