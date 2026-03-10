# Sentiment Analysis of Product Reviews using Transfer Learning with Hugging Face API

This project demonstrates how to perform transfer learning by fine-tuning a pre-trained language model on a dataset of product reviews (sentiment analysis) using the Hugging Face `transformers` library. It also includes a modern web-based UI built with Gradio for analyzing sentiment interactively.

## 🚀 Features
- **Transfer Learning Setup:** Fine-tunes a miniature Transformer (BERT) on sentiment data.
- **Hugging Face Trainer API:** Highly optimized training loop.
- **Interactive UI:** User-friendly Gradio web interface to test the fine-tuned model (or fallback to an existing HF model if not trained yet).

## 🛠️ Installation

Create a virtual environment (optional but recommended) and install the dependencies:

```bash
pip install -r requirements.txt
```

## 🧠 Training the Model (Transfer Learning)
If you want to train (fine-tune) the model from scratch on your own machine using a small subset of data, run:

```bash
python train.py
```
This script will download a small subset of reviews, fine-tune `prajjwal1/bert-tiny` (for faster prototyping), evaluate it, and save the resultant model as `./fine_tuned_sentiment_model`. 

*Note: You can easily switch the model and dataset in `train.py` to something larger like `bert-base-uncased` and `amazon_polarity` for production usage.*

## 🌐 Running the Web Application
Launch the Gradio interface to test the model:

```bash
python app.py
```

The app will use the locally trained model if it exists, otherwise it will fallback to using Hugging Face's default pre-trained sentiment analysis model via its pipeline API.
