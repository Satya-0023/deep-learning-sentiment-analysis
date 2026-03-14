# Model Files Directory

This directory contains the trained LSTM model and tokenizer files required for the Streamlit application.

## Required Files

Place the following files in this directory:

1. **sentiment_model.h5** - The trained Keras LSTM model
2. **tokenizer.pkl** - The fitted tokenizer for text preprocessing

## How to Obtain These Files

These files are generated when you run the training notebook (`notebooks/sentiment_training.ipynb`).

Alternatively, if you trained the model in Google Colab, download:

- `sentiment_model.h5` from your Colab session
- `tokenizer.pkl` from your Colab session

And place them in this directory.

## File Descriptions

### sentiment_model.h5

- Keras model file in HDF5 format
- Contains the trained LSTM neural network weights
- Architecture: Embedding → LSTM → Dense → Sigmoid

### tokenizer.pkl

- Pickled Keras Tokenizer object
- Contains the vocabulary learned during training
- Used to convert text into numerical sequences
