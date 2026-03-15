"""
Sentiment Analysis Web Application using LSTM
Streamlit interface for predicting sentiment of movie reviews.

Pipeline: User Input → Tokenizer → Sequence → Padding (200) → LSTM Model → Prediction
"""

import os
# Must be set BEFORE importing streamlit or tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from tensorflow.keras.utils import pad_sequences

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
    Load the pre-trained LSTM model from sentiment_model.h5.
    Uses h5py directly to bypass Keras 3's broken H5 weight loader
    which crashes with NoneType.pop on Keras 2 → Keras 3 format mismatches.
    """
    import warnings
    import h5py
    import numpy as np
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

    h5_path = os.path.join(MODEL_DIR, 'sentiment_model.h5')
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model file not found: {h5_path}")

    # Build the exact architecture stored in the H5 file
    model = Sequential([
        Input(shape=(MAX_SEQUENCE_LENGTH,), name='input_layer'),
        Embedding(input_dim=5000, output_dim=128, name='embedding'),
        LSTM(128, dropout=0.2, name='lstm'),
        Dense(1, activation='sigmoid', name='dense')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load weights directly with h5py — no Keras .load_weights() involved.
    # Keras 3's loader internally calls .pop() on dicts that are None
    # in Keras 2 H5 files, causing hard crashes. h5py + set_weights() is safe.
    with h5py.File(h5_path, 'r') as f:
        mw = f['model_weights']
        # Embedding
        emb_w = np.array(mw['embedding']['sequential']['embedding']['embeddings'])
        model.get_layer('embedding').set_weights([emb_w])
        # LSTM: kernel, recurrent_kernel, bias
        lc = mw['lstm']['sequential']['lstm']['lstm_cell']
        model.get_layer('lstm').set_weights([
            np.array(lc['kernel']),
            np.array(lc['recurrent_kernel']),
            np.array(lc['bias'])
        ])
        # Dense: kernel, bias
        d = mw['dense']['sequential']['dense']
        model.get_layer('dense').set_weights([
            np.array(d['kernel']),
            np.array(d['bias'])
        ])

    return model, "h5py_direct"




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
    Get prediction from model.
    Uses model.predict() to avoid triggering TF 2.20's while_loop code path
    (which LSTM's recurrent_dropout activates and calls .pop() on a None TensorArray).
    """
    return float(model.predict(padded_input, verbose=0)[0][0])


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

    st.warning(
        "⚠️ **This model is trained exclusively on IMDB Movie Reviews.** "
        "It is designed to analyse **movie review text only**. "
        "Entering non-movie text (product reviews, restaurant reviews, general sentences) "
        "may produce **inaccurate or unexpected results.**"
    )
    st.info(
        "💡 **For best results:** Write a detailed movie review (15+ words) "
        "using clear positive or negative language."
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

    # Session state initialization
    if 'review_text' not in st.session_state:
        st.session_state.review_text = ''

    col1, col2 = st.columns([2, 1])

    # col2 is rendered FIRST so the buttons can update session_state
    # BEFORE the text_area in col1 reads it via value=.
    with col2:
        st.subheader("📋 Examples")
        example = st.selectbox(
            "Select example:",
            ["-- Select --"] + list(SAMPLE_REVIEWS.keys()),
            label_visibility="collapsed"
        )

        if example != "-- Select --":
            if st.button("Use This Example", use_container_width=True):
                st.session_state.review_text = SAMPLE_REVIEWS[example]

        if st.session_state.review_text:
            if st.button("Clear", use_container_width=True):
                st.session_state.review_text = ''

    with col1:
        st.subheader("📝 Enter Movie Review")
        # Use value= (not key=) so Streamlit never locks this session state key.
        # The widget reads the current value; buttons above have already updated it.
        user_input = st.text_area(
            "Review",
            value=st.session_state.review_text,
            placeholder="Example: This movie was amazing! The acting was superb and the story was engaging...",
            height=150,
            label_visibility="collapsed"
        )


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
            # FIX 1: sigmoid can return values microscopically outside [0, 1]
            # due to floating-point rounding. Clip before passing to st.progress
            # to prevent a hard StreamlitAPIException crash.
            st.progress(float(np.clip(raw_score, 0.0, 1.0)))
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
