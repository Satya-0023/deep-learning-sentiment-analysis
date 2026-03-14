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

# Sample movie reviews for demonstration
SAMPLE_REVIEWS = {
    "Positive Example 1": "This movie was absolutely fantastic! The acting was superb, the plot kept me engaged throughout, and the cinematography was breathtaking. Highly recommend!",
    "Positive Example 2": "A masterpiece of storytelling. The director did an amazing job bringing the characters to life. I laughed, I cried, and I left the theater feeling inspired.",
    "Negative Example 1": "Terrible waste of time. The plot made no sense, the acting was wooden, and I nearly fell asleep halfway through. Save your money.",
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
    """Preprocess input text for model prediction."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequence


def predict_sentiment(text, model, tokenizer):
    """Generate sentiment prediction for the given text."""
    processed_text = preprocess_text(text, tokenizer)
    prediction = model.predict(processed_text, verbose=0)
    score = float(prediction[0][0])
    return score


def get_confidence_level(score):
    """Determine confidence level based on how far from 0.5 the score is."""
    distance_from_threshold = abs(score - 0.5)
    if distance_from_threshold < 0.1:
        return "Low", "warning"
    elif distance_from_threshold < 0.25:
        return "Medium", "info"
    else:
        return "High", "success"


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
                user_input = SAMPLE_REVIEWS[selected_example]
                st.rerun()

    # Show selected example in text area
    if selected_example != "-- Select --" and not user_input:
        user_input = SAMPLE_REVIEWS[selected_example]
        st.text_area("Selected Review:", value=user_input, height=100, disabled=True)

    # Prediction button
    st.markdown("")
    predict_clicked = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)

    if predict_clicked:
        if user_input and user_input.strip():
            # Check input quality
            word_count = len(user_input.split())

            with st.spinner("Analyzing sentiment..."):
                score = predict_sentiment(user_input, model, tokenizer)

            st.divider()
            st.subheader("📊 Analysis Results")

            # Warning for short input
            if word_count < 5:
                st.warning("⚠️ Your input is very short. For better accuracy, please enter a longer movie review (at least 10-20 words).")

            # Determine sentiment and confidence
            is_positive = score > 0.5
            confidence_pct = score if is_positive else (1 - score)
            confidence_level, confidence_type = get_confidence_level(score)

            # Results display
            col1, col2, col3 = st.columns(3)

            with col1:
                sentiment_emoji = "😊" if is_positive else "😞"
                sentiment_text = "POSITIVE" if is_positive else "NEGATIVE"
                st.metric(
                    label="Sentiment",
                    value=f"{sentiment_emoji} {sentiment_text}"
                )

            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence_pct:.1%}"
                )

            with col3:
                st.metric(
                    label="Confidence Level",
                    value=confidence_level
                )

            # Visual progress bar
            st.markdown("**Sentiment Score:**")
            st.progress(score)

            col_neg, col_pos = st.columns(2)
            with col_neg:
                st.caption("← Negative (0%)")
            with col_pos:
                st.caption("Positive (100%) →")

            # Interpretation message
            st.markdown("---")
            st.markdown("**Interpretation:**")

            if confidence_level == "Low":
                st.warning(f"""
                🤔 **Uncertain Prediction**

                The model is **not confident** about this prediction (score: {score:.2%}).

                This could happen because:
                - The text is too short or unclear
                - The text is not related to movie reviews
                - The sentiment is genuinely mixed/neutral

                **Tip:** Try entering a longer, more detailed movie review for better results.
                """)
            elif is_positive:
                st.success(f"""
                ✅ **Positive Sentiment Detected**

                The model predicts this is a **positive movie review** with **{confidence_pct:.1%}** confidence.

                The review appears to express favorable opinions about the movie.
                """)
            else:
                st.error(f"""
                ❌ **Negative Sentiment Detected**

                The model predicts this is a **negative movie review** with **{confidence_pct:.1%}** confidence.

                The review appears to express unfavorable opinions about the movie.
                """)

        else:
            st.warning("⚠️ Please enter a movie review to analyze.")

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

        ### Limitations

        - **Domain-Specific:** Trained only on movie reviews, may not work well on other types of text
        - **English Only:** Only understands English language text
        - **Grammar Sensitive:** Works better with grammatically correct sentences
        - **Context:** May miss sarcasm, irony, or complex sentiment expressions

        ### Best Practices

        1. Enter complete movie reviews (not just single words)
        2. Use proper English grammar
        3. Provide context (mention the movie, acting, plot, etc.)
        4. Longer reviews (50+ words) typically give more accurate results
        """)

    st.markdown("---")
    st.caption("🎓 Academic Deep Learning Project | Built with TensorFlow & Streamlit")


if __name__ == "__main__":
    main()
