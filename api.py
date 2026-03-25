"""
Flask REST API for Sentiment Analysis System
Exposes the LSTM Deep Learning model as a RESTful service.
"""

import os
# Must be set BEFORE importing tensorflow to handle Keras 3 compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import tensorflow as tf
import h5py

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from tensorflow.keras.utils import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for frontend integration
CORS(app)

# Constants
MAX_SEQUENCE_LENGTH = 200
MODEL_DIR = "model"
TOKENIZER_PATH = "model/tokenizer.pkl"

def load_trained_model():
    """
    Load the pre-trained LSTM model from sentiment_model.h5 using h5py directly.
    Provides bulletproof loading across Keras versions.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    h5_path = os.path.join(MODEL_DIR, 'sentiment_model.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model file not found: {h5_path}")

    # Build the exact architecture
    model = Sequential([
        Input(shape=(MAX_SEQUENCE_LENGTH,), name='input_layer'),
        Embedding(input_dim=5000, output_dim=128, name='embedding'),
        LSTM(128, dropout=0.2, name='lstm'),
        Dense(1, activation='sigmoid', name='dense')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load weights directly with h5py
    with h5py.File(h5_path, 'r') as f:
        mw = f['model_weights']
        
        # Embedding
        emb_w = np.array(mw['embedding']['sequential']['embedding']['embeddings'])
        model.get_layer('embedding').set_weights([emb_w])
        
        # LSTM
        lc = mw['lstm']['sequential']['lstm']['lstm_cell']
        model.get_layer('lstm').set_weights([
            np.array(lc['kernel']),
            np.array(lc['recurrent_kernel']),
            np.array(lc['bias'])
        ])
        
        # Dense
        d = mw['dense']['sequential']['dense']
        model.get_layer('dense').set_weights([
            np.array(d['kernel']),
            np.array(d['bias'])
        ])

    return model

def load_tokenizer():
    """Load the fitted tokenizer from disk."""
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def preprocess_text(text, tokenizer):
    """Convert text to padded sequence."""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return padded

# Global initialization (Lazy loading can be used, but global is fine for this scale)
try:
    model = load_trained_model()
    tokenizer = load_tokenizer()
    print("✅ Model and Tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/tokenizer: {e}")
    model = None
    tokenizer = None

@app.route("/predict", methods=["POST"])
def predict():
    """
    REST API Endpoint: POST /predict
    Accepts JSON: {"text": "movie review here"}
    Returns JSON: Sentiment prediction and confidence score
    """
    if model is None or tokenizer is None:
        return jsonify({"error": "Model failed to load on the server."}), 500

    # Ensure request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get("text", "")

    # Handle empty input
    if not text or text.strip() == "":
        return jsonify({"error": "Empty input text provided"}), 400

    # Prediction Pipeline
    try:
        # Preprocess
        padded = preprocess_text(text, tokenizer)
        
        # Predict
        prediction = float(model.predict(padded, verbose=0)[0][0])
        raw_score = float(np.clip(prediction, 0.0, 1.0))

        # Determine sentiment with confidence logic
        if raw_score >= 0.6:
            sentiment = "Positive 😊"
            confidence = raw_score * 100
        elif raw_score <= 0.4:
            sentiment = "Negative 😞"
            confidence = (1 - raw_score) * 100
        else:
            sentiment = "Uncertain ⚠️"
            confidence = 50 + (0.5 - abs(raw_score - 0.5)) * 100

        # Formulate Response
        response = {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "raw_score": round(raw_score, 4)
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint."""
    return jsonify({"status": "API is running", "model_loaded": model is not None}), 200

if __name__ == "__main__":
    # Start the Flask development server on port 5000
    print("🚀 Starting Sentiment Analysis API on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
