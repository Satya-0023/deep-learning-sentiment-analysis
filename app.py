"""
Sentiment Analysis Web Application using LSTM
Streamlit interface for predicting sentiment of movie reviews.

Pipeline: User Input → Tokenizer → Sequence → Padding (200) → LSTM Model → Prediction
"""

import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_SEQUENCE_LENGTH = 200
MODEL_DIR = "model"  # SavedModel format directory
TOKENIZER_PATH = "model/tokenizer.pkl"

# Sample reviews for demo
SAMPLE_REVIEWS = {
    "Positive Example 1": "This movie was amazing and inspiring! The acting was superb, the plot kept me engaged throughout, and the cinematography was breathtaking.",
    "Positive Example 2": "A masterpiece! I loved every minute of it. The director did an incredible job and the performances were outstanding.",
    "Negative Example 1": "This movie was terrible and boring. The plot made no sense, the acting was wooden, and I nearly fell asleep.",
    "Negative Example 2": "Worst film I've ever seen. Complete waste of time and money. The story was predictable and the characters were annoying."
}


@st.cache_resource
def load_trained_model():
    """
    Load the pre-trained LSTM model.
    Handles both Keras 2 and Keras 3 compatibility.
    """
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Check if SavedModel exists
    savedmodel_path = MODEL_DIR

    if os.path.exists(os.path.join(savedmodel_path, 'saved_model.pb')):
        # Method 1: Try Keras 3 TFSMLayer (for SavedModel format)
        try:
            from tensorflow import keras
            model = keras.layers.TFSMLayer(savedmodel_path, call_endpoint='serving_default')
            return model, "tfsm_layer"
        except Exception as e1:
            pass

        # Method 2: Try standard load_model (Keras 2 style)
        try:
            model = tf.keras.models.load_model(savedmodel_path)
            return model, "savedmodel"
        except Exception as e2:
            pass

        # Method 3: Try with tf.saved_model.load
        try:
            model = tf.saved_model.load(savedmodel_path)
            return model, "tf_saved_model"
        except Exception as e3:
            pass

    # Method 4: Try loading .keras or .h5 file
    for ext in ['.keras', '.h5']:
        h5_path = os.path.join(MODEL_DIR, f'sentiment_model{ext}')
        if os.path.exists(h5_path):
            try:
                model = tf.keras.models.load_model(h5_path, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model, f"keras_file({ext})"
            except:
                pass

    raise FileNotFoundError("No valid model found in model/ directory")


@st.cache_resource
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


def get_prediction(model, padded_input):
    """
    Get prediction from model, handling different model types.
    TFSMLayer returns dict, regular model returns array.
    """
    result = model(padded_input)

    # Handle TFSMLayer output (returns dictionary)
    if isinstance(result, dict):
        # Get the first output key
        key = list(result.keys())[0]
        return float(result[key].numpy()[0][0])
    # Handle regular model output
    elif hasattr(result, 'numpy'):
        return float(result.numpy()[0][0])
    else:
        return float(result[0][0])


def verify_model(model, tokenizer):
    """Verify model produces different outputs for positive vs negative text."""
    pos_text = "This movie was amazing fantastic wonderful excellent masterpiece"
    neg_text = "This movie was terrible awful horrible worst disaster"

    pos_padded = preprocess_text(pos_text, tokenizer)
    neg_padded = preprocess_text(neg_text, tokenizer)

    pos_pred = get_prediction(model, pos_padded)
    neg_pred = get_prediction(model, neg_padded)

    # Model should give higher score for positive text
    is_working = pos_pred > neg_pred and abs(pos_pred - neg_pred) > 0.1
    return is_working, pos_pred, neg_pred


def predict_sentiment(text, model, tokenizer):
    """Generate sentiment prediction."""
    padded = preprocess_text(text, tokenizer)
    prediction = get_prediction(model, padded)

    # Determine sentiment with uncertain zone
    if prediction >= 0.6:
        sentiment = "Positive 😊"
        confidence = prediction * 100
    elif prediction <= 0.4:
        sentiment = "Negative 😞"
        confidence = (1 - prediction) * 100
    else:
        sentiment = "Uncertain ⚠️"
        confidence = 50 + (0.5 - abs(prediction - 0.5)) * 100

    return sentiment, confidence, prediction


def main():
    st.set_page_config(
        page_title="Movie Review Sentiment Analysis",
        page_icon="🎬",
        layout="centered"
    )

    st.title("🎬 Movie Review Sentiment Analysis")
    st.caption("Deep Learning Sentiment Analysis using LSTM")

    st.info(
        "**Note:** This model is trained on **IMDB Movie Reviews**. "
        "For best results, enter detailed movie reviews with proper grammar."
    )

    # Load model and tokenizer
    try:
        model, load_method = load_trained_model()
        tokenizer = load_tokenizer()

        # Verify model is working
        is_working, pos_score, neg_score = verify_model(model, tokenizer)

        if is_working:
            st.success(f"✅ Model loaded successfully! (Method: {load_method})")
            st.caption(f"Verification - Positive: {pos_score:.4f} | Negative: {neg_score:.4f}")
        else:
            st.warning(f"""
            ⚠️ **Model loaded but predictions may be inaccurate.**

            - Load method: `{load_method}`
            - Test positive score: {pos_score:.4f}
            - Test negative score: {neg_score:.4f}
            """)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return

    st.divider()

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Enter Movie Review")
        user_input = st.text_area(
            "Review",
            placeholder="Example: This movie was amazing! The acting was superb and the story was engaging...",
            height=150,
            label_visibility="collapsed"
        )

    with col2:
        st.subheader("📋 Examples")
        example = st.selectbox(
            "Select example:",
            ["-- Select --"] + list(SAMPLE_REVIEWS.keys()),
            label_visibility="collapsed"
        )

        if example != "-- Select --":
            if st.button("Use This Example", use_container_width=True):
                st.session_state['example_text'] = SAMPLE_REVIEWS[example]
                st.rerun()

    # Use example if selected
    if 'example_text' in st.session_state:
        user_input = st.session_state['example_text']
        st.text_area("Selected:", value=user_input, height=80, disabled=True)
        if st.button("Clear", use_container_width=True):
            del st.session_state['example_text']
            st.rerun()

    # Predict button
    if st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True):
        if not user_input or not user_input.strip():
            st.warning("⚠️ Please enter a movie review.")
        else:
            word_count = len(user_input.split())
            if word_count < 5:
                st.warning("⚠️ Input is very short. Results may be inaccurate.")

            with st.spinner("Analyzing..."):
                sentiment, confidence, raw_score = predict_sentiment(user_input, model, tokenizer)

            st.divider()
            st.subheader("📊 Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", sentiment)
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col3:
                st.metric("Raw Score", f"{raw_score:.4f}")

            # Score bar
            st.progress(raw_score)
            c1, c2, c3 = st.columns(3)
            c1.caption("← Negative (0)")
            c2.caption("Uncertain")
            c3.caption("Positive (1) →")

            # Interpretation
            if "Positive" in sentiment:
                st.success(f"✅ The model predicts this is a **positive** review ({confidence:.1f}% confidence).")
            elif "Negative" in sentiment:
                st.error(f"❌ The model predicts this is a **negative** review ({confidence:.1f}% confidence).")
            else:
                st.warning(f"⚠️ The model is **uncertain** about this review. Try a longer, clearer review.")

    # Footer
    st.divider()
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Model:** LSTM Neural Network
        **Dataset:** IMDB 50K Movie Reviews
        **Accuracy:** ~88%

        **Prediction Zones:**
        - Score > 0.6 → Positive
        - Score < 0.4 → Negative
        - 0.4 - 0.6 → Uncertain
        """)

    st.caption("🎓 Academic Deep Learning Project | TensorFlow + Streamlit")


if __name__ == "__main__":
    main()
