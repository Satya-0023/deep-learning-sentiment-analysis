"""
Sentiment Analysis Web Application using LSTM
Streamlit interface for predicting sentiment of movie reviews.

Pipeline: User Input → Tokenizer → Sequence → Padding (200) → LSTM Model → Prediction
"""

import streamlit as st
import numpy as np
import pickle
import h5py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_SEQUENCE_LENGTH = 200
MODEL_PATH_H5 = "model/sentiment_model.h5"
MODEL_PATH_KERAS = "model/sentiment_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"

# Sample reviews for demo
SAMPLE_REVIEWS = {
    "Positive Example 1": "This movie was amazing and inspiring! The acting was superb, the plot kept me engaged throughout, and the cinematography was breathtaking.",
    "Positive Example 2": "A masterpiece! I loved every minute of it. The director did an incredible job and the performances were outstanding.",
    "Negative Example 1": "This movie was terrible and boring. The plot made no sense, the acting was wooden, and I nearly fell asleep.",
    "Negative Example 2": "Worst film I've ever seen. Complete waste of time and money. The story was predictable and the characters were annoying."
}


def build_model():
    """Build the LSTM model architecture."""
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=200, name='embedding'),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm'),
        Dense(1, activation='sigmoid', name='dense')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_weights_from_h5(model, h5_path):
    """
    Extract weights from H5 file and load into model.
    This handles Keras 3.x to Keras 2.x compatibility issues.
    """
    with h5py.File(h5_path, 'r') as f:
        # Print structure for debugging
        def print_structure(name, obj):
            pass  # Uncomment below for debugging
            # print(name)
        # f.visititems(print_structure)

        # Try different H5 structures (Keras saves in different formats)

        # Structure 1: model_weights/layer_name/layer_name/weight_name
        if 'model_weights' in f:
            weights_group = f['model_weights']
        else:
            weights_group = f

        # Load Embedding weights
        embedding_loaded = False
        for path in ['embedding/embedding/embeddings:0', 'embedding/embeddings:0',
                     'embedding_1/embedding_1/embeddings:0', 'embedding_1/embeddings:0']:
            try:
                parts = path.split('/')
                group = weights_group
                for part in parts[:-1]:
                    if part in group:
                        group = group[part]
                if parts[-1] in group:
                    emb_weights = np.array(group[parts[-1]])
                    model.layers[0].set_weights([emb_weights])
                    embedding_loaded = True
                    break
            except:
                continue

        # Load LSTM weights
        lstm_loaded = False
        for prefix in ['lstm', 'lstm_1']:
            try:
                if prefix in weights_group:
                    lstm_group = weights_group[prefix]
                    if prefix in lstm_group:
                        lstm_group = lstm_group[prefix]

                    kernel = None
                    recurrent = None
                    bias = None

                    for key in lstm_group.keys():
                        if 'kernel' in key and 'recurrent' not in key:
                            kernel = np.array(lstm_group[key])
                        elif 'recurrent_kernel' in key:
                            recurrent = np.array(lstm_group[key])
                        elif 'bias' in key:
                            bias = np.array(lstm_group[key])

                    if kernel is not None and recurrent is not None and bias is not None:
                        model.layers[1].set_weights([kernel, recurrent, bias])
                        lstm_loaded = True
                        break
            except:
                continue

        # Load Dense weights
        dense_loaded = False
        for prefix in ['dense', 'dense_1']:
            try:
                if prefix in weights_group:
                    dense_group = weights_group[prefix]
                    if prefix in dense_group:
                        dense_group = dense_group[prefix]

                    kernel = None
                    bias = None

                    for key in dense_group.keys():
                        if 'kernel' in key:
                            kernel = np.array(dense_group[key])
                        elif 'bias' in key:
                            bias = np.array(dense_group[key])

                    if kernel is not None and bias is not None:
                        model.layers[2].set_weights([kernel, bias])
                        dense_loaded = True
                        break
            except:
                continue

        return embedding_loaded, lstm_loaded, dense_loaded


@st.cache_resource
def load_trained_model():
    """
    Load the pre-trained LSTM model.
    Handles Keras version compatibility issues by manually loading weights.
    """
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Determine which model file exists
    if os.path.exists(MODEL_PATH_KERAS):
        model_path = MODEL_PATH_KERAS
    elif os.path.exists(MODEL_PATH_H5):
        model_path = MODEL_PATH_H5
    else:
        raise FileNotFoundError("No model file found. Please add sentiment_model.h5 or sentiment_model.keras to model/ folder.")

    # Method 1: Try direct loading (works if Keras versions match)
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model, "direct_load"
    except Exception as e1:
        pass

    # Method 2: Try with compile=False and safe_mode=False
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False, safe_mode=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model, "safe_mode_false"
    except Exception as e2:
        pass

    # Method 3: Build model and load weights from H5 file
    try:
        model = build_model()
        model.build(input_shape=(None, 200))

        # Find the H5 file (could be .h5 or .keras)
        h5_path = MODEL_PATH_H5 if os.path.exists(MODEL_PATH_H5) else MODEL_PATH_KERAS

        emb_loaded, lstm_loaded, dense_loaded = load_weights_from_h5(model, h5_path)

        if emb_loaded and lstm_loaded and dense_loaded:
            return model, "manual_weights"
        else:
            return model, f"partial_weights(emb={emb_loaded},lstm={lstm_loaded},dense={dense_loaded})"
    except Exception as e3:
        pass

    # Method 4: Return untrained model as last resort
    model = build_model()
    model.build(input_shape=(None, 200))
    return model, "untrained_fallback"


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


def verify_model(model, tokenizer):
    """Verify model produces different outputs for positive vs negative text."""
    pos_text = "This movie was amazing fantastic wonderful excellent"
    neg_text = "This movie was terrible awful horrible worst"

    pos_pred = float(model.predict(preprocess_text(pos_text, tokenizer), verbose=0)[0][0])
    neg_pred = float(model.predict(preprocess_text(neg_text, tokenizer), verbose=0)[0][0])

    # Model should give higher score for positive text
    is_working = pos_pred > neg_pred and abs(pos_pred - neg_pred) > 0.1
    return is_working, pos_pred, neg_pred


def predict_sentiment(text, model, tokenizer):
    """Generate sentiment prediction."""
    padded = preprocess_text(text, tokenizer)
    prediction = float(model.predict(padded, verbose=0)[0][0])

    # Determine sentiment with uncertain zone
    if prediction >= 0.6:
        sentiment = "Positive 😊"
        confidence = prediction * 100
    elif prediction <= 0.4:
        sentiment = "Negative 😞"
        confidence = (1 - prediction) * 100
    else:
        sentiment = "Uncertain ⚠️"
        confidence = (1 - abs(prediction - 0.5) * 2) * 100

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
        else:
            st.warning(f"""
            ⚠️ **Model loaded but predictions may be inaccurate.**

            - Load method: `{load_method}`
            - Test positive score: {pos_score:.4f}
            - Test negative score: {neg_score:.4f}

            **Fix:** Run `pip install --upgrade tensorflow` or re-export model from Colab.
            """)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("""
        **Troubleshooting:**
        1. Ensure `sentiment_model.h5` is in the `model/` folder
        2. Run: `pip install --upgrade tensorflow`
        3. Or re-save model in Colab: `model.save('sentiment_model.h5', save_format='h5')`
        """)
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
