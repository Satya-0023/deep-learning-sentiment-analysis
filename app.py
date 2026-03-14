"""
Sentiment Analysis Web Application using LSTM
This Streamlit application provides a user-friendly interface for sentiment prediction
on movie reviews using a pre-trained LSTM deep learning model.
"""

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_SEQUENCE_LENGTH = 200
MODEL_PATH = 'model/sentiment_model.h5'
TOKENIZER_PATH = 'model/tokenizer.pkl'


@st.cache_resource
def load_trained_model():
    """
    Load the pre-trained LSTM model from disk.
    Uses Streamlit's cache to avoid reloading on every interaction.
    Handles Keras version compatibility issues.

    Returns:
        model: Loaded Keras LSTM model
    """
    import warnings
    warnings.filterwarnings('ignore')

    # Try multiple loading approaches for version compatibility
    try:
        # Approach 1: Standard loading with compile=False
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e1:
        try:
            # Approach 2: Try with safe_mode=False (newer Keras versions)
            model = load_model(MODEL_PATH, compile=False, safe_mode=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e2:
            # Approach 3: Rebuild model architecture and load weights
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, LSTM, Dense

            model = Sequential([
                Embedding(input_dim=5000, output_dim=128, input_length=200),
                LSTM(128, dropout=0.2, recurrent_dropout=0.2),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Try to load weights from H5 file
            try:
                model.load_weights(MODEL_PATH)
            except:
                # If weights loading fails, try loading full model with legacy format
                import h5py
                with h5py.File(MODEL_PATH, 'r') as f:
                    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)

            return model


@st.cache_resource
def load_tokenizer():
    """
    Load the fitted tokenizer from disk.
    The tokenizer converts text into numerical sequences based on the vocabulary
    learned during training.

    Returns:
        tokenizer: Loaded Keras Tokenizer object
    """
    with open(TOKENIZER_PATH, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer


def preprocess_text(text, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    """
    Preprocess input text for model prediction.

    Steps:
    1. Convert text to sequences using the tokenizer
    2. Pad sequences to ensure uniform length

    Args:
        text (str): Raw input text from user
        tokenizer: Fitted Keras Tokenizer
        max_length (int): Maximum sequence length for padding

    Returns:
        numpy.ndarray: Padded sequence ready for model input
    """
    # Convert text to sequence of integers
    sequences = tokenizer.texts_to_sequences([text])

    # Pad sequence to fixed length
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    return padded_sequence


def predict_sentiment(text, model, tokenizer):
    """
    Generate sentiment prediction for the given text.

    Args:
        text (str): Input text for sentiment analysis
        model: Loaded LSTM model
        tokenizer: Loaded tokenizer

    Returns:
        tuple: (prediction_score, sentiment_label)
    """
    # Preprocess the input text
    processed_text = preprocess_text(text, tokenizer)

    # Generate prediction
    prediction = model.predict(processed_text, verbose=0)
    score = prediction[0][0]

    # Determine sentiment label
    if score > 0.5:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return score, sentiment


def main():
    """
    Main function to run the Streamlit application.
    """
    # Page configuration
    st.set_page_config(
        page_title="Sentiment Analysis - LSTM",
        page_icon="🎬",
        layout="centered"
    )

    # Application title and description
    st.title("🎬 Sentiment Analysis using LSTM")
    st.markdown("""
    This application uses a **Deep Learning LSTM model** trained on the IMDB movie reviews dataset
    to predict the sentiment of text input.

    Enter a movie review below and click **Predict Sentiment** to see the results!
    """)

    st.divider()

    # Load model and tokenizer
    try:
        model = load_trained_model()
        tokenizer = load_tokenizer()
        st.success("✅ Model and tokenizer loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model or tokenizer: {str(e)}")
        st.info("Please ensure 'sentiment_model.h5' and 'tokenizer.pkl' are in the 'model/' directory.")
        return

    # Text input area
    st.subheader("📝 Enter Your Movie Review")
    user_input = st.text_area(
        label="Movie Review",
        placeholder="Type or paste your movie review here...",
        height=150,
        label_visibility="collapsed"
    )

    # Prediction button
    if st.button("🔮 Predict Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                score, sentiment = predict_sentiment(user_input, model, tokenizer)

            # Display results
            st.divider()
            st.subheader("📊 Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="Sentiment", value=sentiment)

            with col2:
                st.metric(label="Confidence Score", value=f"{score:.2%}")

            # Progress bar for visualization
            st.progress(float(score))

            # Additional information
            if score > 0.5:
                st.success(f"The model predicts this review is **POSITIVE** with {score:.2%} confidence.")
            else:
                st.error(f"The model predicts this review is **NEGATIVE** with {(1-score):.2%} confidence.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

    # Footer
    st.divider()
    st.markdown("""
    ---
    **About this project:**
    - Model: LSTM (Long Short-Term Memory) Neural Network
    - Dataset: IMDB 50K Movie Reviews
    - Framework: TensorFlow/Keras + Streamlit
    """)


if __name__ == "__main__":
    main()
