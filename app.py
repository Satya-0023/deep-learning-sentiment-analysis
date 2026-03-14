"""
Sentiment Analysis Web Application using LSTM
This Streamlit application provides a user-friendly interface for sentiment prediction
on movie reviews using a pre-trained LSTM deep learning model.

Prediction Pipeline:
User Input → Tokenizer → Text to Sequence → Padding (200 tokens) → LSTM Model → Sentiment
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

# Sample movie reviews for demonstration
SAMPLE_REVIEWS = {
    "Positive Example 1": "This movie was amazing and inspiring! The acting was superb, the plot kept me engaged throughout, and the cinematography was breathtaking. Highly recommend!",
    "Positive Example 2": "A masterpiece of storytelling. The director did an amazing job bringing the characters to life. I laughed, I cried, and I left the theater feeling inspired.",
    "Negative Example 1": "This movie was terrible and boring. The plot made no sense, the acting was wooden, and I nearly fell asleep halfway through. Save your money.",
    "Negative Example 2": "I was really disappointed with this film. The story was predictable, the characters were one-dimensional, and the pacing was painfully slow."
}


@st.cache_resource
def load_trained_model():
    """
    Load the pre-trained LSTM model from disk.
    Handles Keras version compatibility issues.
    """
    import warnings
    warnings.filterwarnings('ignore')

    try:
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e1:
        try:
            model = load_model(MODEL_PATH, compile=False, safe_mode=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e2:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, LSTM, Dense

            model = Sequential([
                Embedding(input_dim=5000, output_dim=128, input_length=200),
                LSTM(128, dropout=0.2, recurrent_dropout=0.2),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            try:
                model.load_weights(MODEL_PATH)
            except:
                import h5py
                with h5py.File(MODEL_PATH, 'r') as f:
                    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)

            return model


@st.cache_resource
def load_tokenizer():
    """Load the fitted tokenizer from disk."""
    with open(TOKENIZER_PATH, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer


def preprocess_text(text, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    """
    Preprocess input text for model prediction.

    Pipeline: Text → Tokenizer → Sequence → Padding (200 tokens)
    """
    # Convert text to sequence of integers
    sequences = tokenizer.texts_to_sequences([text])
    # Pad sequence to fixed length of 200 tokens
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequence


def predict_sentiment(text, model, tokenizer):
    """
    Generate sentiment prediction for the given text.

    Returns:
        confidence (float): Model output probability (0.0 to 1.0)
            - Values closer to 1.0 indicate positive sentiment
            - Values closer to 0.0 indicate negative sentiment
    """
    # Preprocess: Tokenize and pad the input text
    padded_sequence = preprocess_text(text, tokenizer)

    # Get model prediction
    prediction = model.predict(padded_sequence, verbose=0)

    # Extract confidence score (probability)
    confidence = float(prediction[0][0])

    return confidence


def get_sentiment_label(confidence):
    """
    Determine sentiment label based on confidence score.

    Uses an uncertain zone (0.4 - 0.6) to avoid misleading predictions
    near the decision boundary.

    Args:
        confidence (float): Model output probability (0.0 to 1.0)

    Returns:
        tuple: (sentiment_label, emoji)
    """
    if confidence > 0.6:
        return "POSITIVE", "😊"
    elif confidence < 0.4:
        return "NEGATIVE", "😞"
    else:
        return "UNCERTAIN", "⚠️"


def get_confidence_level(confidence):
    """
    Classify confidence level based on prediction probability.

    Args:
        confidence (float): Model output probability (0.0 to 1.0)

    Returns:
        tuple: (confidence_level, display_type)
    """
    if confidence > 0.75 or confidence < 0.25:
        return "High", "success"
    elif confidence > 0.60 or confidence < 0.40:
        return "Medium", "info"
    else:
        return "Low", "warning"


def get_confidence_percent(confidence):
    """
    Convert confidence to display percentage.

    For positive predictions (>0.5): shows confidence as-is
    For negative predictions (<0.5): shows inverted confidence (1 - confidence)
    For uncertain (0.4-0.6): shows distance from 0.5

    Args:
        confidence (float): Model output probability (0.0 to 1.0)

    Returns:
        float: Confidence percentage for display
    """
    if confidence > 0.6:
        # Positive: confidence represents positive probability
        return confidence * 100
    elif confidence < 0.4:
        # Negative: invert to show confidence in negative prediction
        return (1 - confidence) * 100
    else:
        # Uncertain: show how close to 50% (low confidence)
        return (0.5 - abs(confidence - 0.5)) * 200  # Scale uncertainty


def main():
    """Main function to run the Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="Movie Review Sentiment Analysis",
        page_icon="🎬",
        layout="centered"
    )

    # Application title
    st.title("🎬 Movie Review Sentiment Analysis")
    st.caption("Powered by LSTM Deep Learning")

    # Important notice about the model
    st.info("""
    **Important:** This model is trained specifically on **IMDB Movie Reviews**.

    For best results:
    - Enter movie/film reviews or opinions about movies
    - Use complete sentences with proper grammar
    - Longer, detailed reviews give more accurate predictions

    *Note: General sentences or non-movie text may produce inaccurate results.*
    """)

    st.divider()

    # Load model and tokenizer
    try:
        model = load_trained_model()
        tokenizer = load_tokenizer()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure model files are in the 'model/' directory.")
        return

    # Two columns: Input and Examples
    col_input, col_examples = st.columns([2, 1])

    with col_input:
        st.subheader("📝 Enter Movie Review")
        user_input = st.text_area(
            label="Movie Review",
            placeholder="Example: This movie was amazing! The plot was engaging and the actors delivered stellar performances...",
            height=180,
            label_visibility="collapsed"
        )

    with col_examples:
        st.subheader("📋 Try Examples")
        selected_example = st.selectbox(
            "Select a sample review:",
            ["-- Select --"] + list(SAMPLE_REVIEWS.keys()),
            label_visibility="collapsed"
        )

        if selected_example != "-- Select --":
            if st.button("Use This Example", use_container_width=True):
                st.session_state['selected_review'] = SAMPLE_REVIEWS[selected_example]
                st.rerun()

    # Use selected example if available
    if 'selected_review' in st.session_state and not user_input:
        user_input = st.session_state['selected_review']
        st.text_area("Selected Review:", value=user_input, height=100, disabled=True)

    # Prediction button
    st.markdown("")
    predict_clicked = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)

    if predict_clicked:
        if user_input and user_input.strip():
            # Check input quality
            word_count = len(user_input.split())

            with st.spinner("Analyzing sentiment..."):
                # Get model prediction (probability score)
                confidence = predict_sentiment(user_input, model, tokenizer)

            st.divider()
            st.subheader("📊 Analysis Results")

            # Warning for short input
            if word_count < 5:
                st.warning("⚠️ Your input is very short. For better accuracy, please enter a longer movie review (at least 10-20 words).")

            # Determine sentiment label and confidence level
            sentiment_label, sentiment_emoji = get_sentiment_label(confidence)
            confidence_level, confidence_type = get_confidence_level(confidence)
            confidence_percent = get_confidence_percent(confidence)

            # Results display - 3 columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Sentiment",
                    value=f"{sentiment_emoji} {sentiment_label}"
                )

            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence_percent:.1f}%"
                )

            with col3:
                st.metric(
                    label="Confidence Level",
                    value=confidence_level
                )

            # Sentiment Score Bar
            st.markdown("**Sentiment Score:**")
            st.progress(confidence)

            # Score bar labels
            col_neg, col_mid, col_pos = st.columns([1, 1, 1])
            with col_neg:
                st.caption("← Negative (0%)")
            with col_mid:
                st.caption("Uncertain")
            with col_pos:
                st.caption("Positive (100%) →")

            # Raw score display
            st.caption(f"Raw Model Output: {confidence:.4f} ({confidence * 100:.2f}%)")

            # Interpretation message
            st.markdown("---")
            st.markdown("**Interpretation:**")

            if sentiment_label == "UNCERTAIN":
                st.warning(f"""
                🤔 **Uncertain Prediction**

                The model prediction is in the **uncertain zone** (score: {confidence:.2%}).

                This means the model cannot confidently classify the sentiment as positive or negative.

                **Possible reasons:**
                - The text contains mixed sentiments
                - The text is too short or unclear
                - The text is not related to movie reviews
                - The sentiment is genuinely neutral

                **Tip:** Try entering a longer, more detailed movie review with clearer positive or negative language.
                """)
            elif sentiment_label == "POSITIVE":
                if confidence_level == "High":
                    st.success(f"""
                    ✅ **Strong Positive Sentiment**

                    The model is **highly confident** ({confidence_percent:.1f}%) that this is a **positive movie review**.

                    The review clearly expresses favorable opinions about the movie.
                    """)
                else:
                    st.success(f"""
                    ✅ **Positive Sentiment Detected**

                    The model predicts this is a **positive movie review** with **{confidence_percent:.1f}%** confidence.

                    The review appears to express favorable opinions about the movie.
                    """)
            else:  # NEGATIVE
                if confidence_level == "High":
                    st.error(f"""
                    ❌ **Strong Negative Sentiment**

                    The model is **highly confident** ({confidence_percent:.1f}%) that this is a **negative movie review**.

                    The review clearly expresses unfavorable opinions about the movie.
                    """)
                else:
                    st.error(f"""
                    ❌ **Negative Sentiment Detected**

                    The model predicts this is a **negative movie review** with **{confidence_percent:.1f}%** confidence.

                    The review appears to express unfavorable opinions about the movie.
                    """)

        else:
            st.warning("⚠️ Please enter a movie review to analyze.")

    # Clear session state button
    if 'selected_review' in st.session_state:
        if st.button("Clear Example", use_container_width=True):
            del st.session_state['selected_review']
            st.rerun()

    # Footer
    st.divider()

    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        ### Model Information

        | Attribute | Value |
        |-----------|-------|
        | **Model Type** | LSTM (Long Short-Term Memory) |
        | **Training Dataset** | IMDB 50K Movie Reviews |
        | **Vocabulary Size** | 5,000 words |
        | **Sequence Length** | 200 tokens |
        | **Test Accuracy** | ~88% |

        ### Prediction Logic

        The model outputs a probability score between 0 and 1:

        | Score Range | Sentiment | Description |
        |-------------|-----------|-------------|
        | **> 0.60** | Positive 😊 | Confident positive prediction |
        | **0.40 - 0.60** | Uncertain ⚠️ | Model is not confident |
        | **< 0.40** | Negative 😞 | Confident negative prediction |

        ### Confidence Levels

        | Confidence Level | Score Range |
        |------------------|-------------|
        | **High** | > 75% or < 25% |
        | **Medium** | 60-75% or 25-40% |
        | **Low** | 40-60% (Uncertain Zone) |

        ### Limitations

        - **Domain-Specific:** Trained only on movie reviews
        - **English Only:** Only understands English text
        - **Grammar Sensitive:** Works better with proper grammar
        - **Context:** May miss sarcasm or irony

        ### Best Practices

        1. Enter complete movie reviews (not single words)
        2. Use proper English grammar
        3. Provide context (movie, acting, plot, etc.)
        4. Longer reviews (50+ words) give better results
        """)

    st.markdown("---")
    st.caption("🎓 Academic Deep Learning Project | Built with TensorFlow & Streamlit")


if __name__ == "__main__":
    main()
