# 🎬 Sentiment Analysis using LSTM Deep Learning

A complete end-to-end machine learning project for sentiment analysis on movie reviews using Long Short-Term Memory (LSTM) neural networks, deployed with a Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [NLP Preprocessing Pipeline](#-nlp-preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [Technologies Used](#-technologies-used)
- [Project Workflow](#-project-workflow)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Example Prediction](#-example-prediction)
- [Future Improvements](#-future-improvements)
- [Contributors](#-contributors)

---

## 🎯 Overview

This project implements a **Deep Learning-based Sentiment Analysis system** that classifies movie reviews as either **Positive** or **Negative**. The system utilizes:

- **LSTM (Long Short-Term Memory)** neural networks to capture sequential dependencies in text data
- **Word Embeddings** to convert textual data into dense vector representations
- **Streamlit** for deploying an interactive web application

The model is trained on the IMDB movie reviews dataset and achieves high accuracy in predicting sentiment from user-provided text.

---

## 🔍 Problem Statement

In the era of digital media, millions of movie reviews are generated daily across various platforms. Manually analyzing these reviews to understand audience sentiment is:

- Time-consuming and labor-intensive
- Prone to human bias and inconsistency
- Not scalable for large volumes of data

**Solution:** An automated sentiment analysis system that can:

- Process text input in real-time
- Provide accurate sentiment predictions
- Scale to handle large volumes of reviews

---

## 📊 Dataset Description

### IMDB 50K Movie Reviews Dataset

| Attribute         | Description                       |
| ----------------- | --------------------------------- |
| **Source**        | IMDB Movie Reviews                |
| **Total Samples** | 50,000 reviews                    |
| **Classes**       | Binary (Positive/Negative)        |
| **Distribution**  | 25,000 Positive + 25,000 Negative |
| **Split**         | 80% Training, 20% Testing         |

The dataset contains highly polar movie reviews, making it ideal for binary sentiment classification tasks.

---

## 🔧 NLP Preprocessing Pipeline

### 1. Text Cleaning

- Removal of HTML tags and special characters
- Converting text to lowercase
- Removing stopwords (optional)

### 2. Tokenization

```
The tokenizer converts text into numerical sequences:

"This movie is great" → [45, 89, 12, 234]

How it works:
1. Build vocabulary from training data
2. Assign unique integer to each word
3. Convert new text using learned vocabulary
4. Out-of-vocabulary words are handled gracefully
```

### 3. Sequence Padding

```
Neural networks require fixed-length input. Padding ensures uniformity:

Original:  [45, 89, 12, 234]           (length: 4)
Padded:    [45, 89, 12, 234, 0, 0...0] (length: 200)

- Short sequences: Padded with zeros
- Long sequences: Truncated to max length
- Max Length: 200 tokens
```

### 4. Why Padding Matters

- **Batch Processing:** Neural networks process data in batches, requiring uniform dimensions
- **Memory Efficiency:** Fixed-size tensors optimize GPU memory usage
- **Model Consistency:** Ensures the model learns patterns regardless of input length

---

## 🏗️ Model Architecture

### LSTM Network Structure

```
┌─────────────────────────────────────────────┐
│            Input Layer (200 tokens)         │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│     Embedding Layer (vocab_size × 128)      │
│   Converts integers to dense vectors        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│           LSTM Layer (128 units)            │
│   Captures sequential dependencies          │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         Dense Layer (64 units, ReLU)        │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│     Output Layer (1 unit, Sigmoid)          │
│   Binary classification (0-1)               │
└─────────────────────────────────────────────┘
```

### How LSTM Captures Sequential Dependencies

LSTM networks are specifically designed for sequential data:

1. **Memory Cell:** Maintains information over long sequences
2. **Forget Gate:** Decides what information to discard
3. **Input Gate:** Decides what new information to store
4. **Output Gate:** Decides what information to output

```
Example: "This movie was not good, but the acting was excellent"

- Traditional models might miss the negation context
- LSTM remembers "not" when processing "good"
- LSTM captures the contrast between "not good" and "excellent"
```

---

## 🛠️ Technologies Used

| Category                    | Technology         |
| --------------------------- | ------------------ |
| **Programming Language**    | Python 3.8+        |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Web Framework**           | Streamlit          |
| **Data Processing**         | NumPy, Pandas      |
| **Machine Learning**        | Scikit-learn       |
| **Development**             | Jupyter Notebook   |
| **Version Control**         | Git, GitHub        |

---

## 📈 Project Workflow

![Workflow Diagram](assets/workflow.svg)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Dataset    │────▶│Preprocessing │────▶│ Tokenization │
│ (IMDB 50K)   │     │  & Cleaning  │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │◀────│  Save Model  │◀────│   Padding    │
│     App      │     │ & Tokenizer  │     │              │
└──────┬───────┘     └──────────────┘     └──────┬───────┘
       │                    ▲                      │
       │                    │                      ▼
       │              ┌──────────────┐     ┌──────────────┐
       │              │  Evaluation  │◀────│ LSTM Model   │
       │              │              │     │   Training   │
       │              └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐
│  Sentiment   │
│  Prediction  │
└──────────────┘
```

### Workflow Steps:

1. **Data Collection:** Load IMDB dataset with 50K movie reviews
2. **Preprocessing:** Clean text, remove noise, normalize
3. **Tokenization:** Convert text to numerical sequences
4. **Padding:** Ensure uniform sequence length (200 tokens)
5. **Model Training:** Train LSTM network on preprocessed data
6. **Evaluation:** Assess model performance on test set
7. **Save Artifacts:** Export model (.h5) and tokenizer (.pkl)
8. **Deployment:** Create Streamlit web interface
9. **Prediction:** Real-time sentiment analysis on user input

---

## 📁 Project Structure

```
sentiment-analysis-project/
│
├── notebooks/
│   └── sentiment_training.ipynb  # Model training notebook
│
├── model/
│   ├── sentiment_model.h5        # Trained LSTM model
│   └── tokenizer.pkl             # Fitted tokenizer
│
├── assets/
│   └── workflow.png              # Project workflow diagram
│
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Satya-0023/deep-learning-sentiment-analysis.git
   cd deep-learning-sentiment-analysis
   ```

2. **Create virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present**

   ```
   Verify that the following files exist:
   - model/sentiment_model.h5
   - model/tokenizer.pkl
   ```

5. **Run the Streamlit application**

   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   ```
   Open your browser and navigate to: http://localhost:8501
   ```

---

## 💡 Example Prediction

### Input

```
"This movie was absolutely fantastic! The acting was superb,
the plot kept me engaged throughout, and the cinematography
was breathtaking. Highly recommend watching it!"
```

### Output

```
Sentiment: Positive 😊
Confidence Score: 94.7%
```

### Another Example

### Input

```
"I was really disappointed with this film. The story was
predictable, the characters were one-dimensional, and the
pacing was terribly slow. Would not recommend."
```

### Output

```
Sentiment: Negative 😞
Confidence Score: 89.2%
```

---

## 🔮 Future Improvements

1. **Model Enhancements**
   - Implement Bidirectional LSTM for better context understanding
   - Add attention mechanism for interpretability
   - Experiment with transformer-based models (BERT, RoBERTa)

2. **Dataset Expansion**
   - Include reviews from multiple sources (Rotten Tomatoes, Amazon)
   - Add multi-class sentiment (Very Positive, Positive, Neutral, Negative, Very Negative)

3. **Feature Additions**
   - Confidence threshold customization
   - Batch prediction for multiple reviews
   - REST API endpoint for integration
   - Sentiment trend analysis over time

4. **Deployment**
   - Docker containerization
   - Cloud deployment (AWS, GCP, Heroku)
   - CI/CD pipeline implementation

5. **User Interface**
   - Dark/Light theme toggle
   - Review history tracking
   - Export predictions to CSV

---

## 👥 Contributors

| Name      | Role      | Contact                                   |
| --------- | --------- | ----------------------------------------- |
| Satya     | Developer | [GitHub](https://github.com/Satya-0023)   |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) for providing the movie reviews data
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the easy-to-use web framework
- Academic advisors and mentors for guidance

---

<p align="center">
  Made with ❤️ for Academic Machine Learning Project
</p>
