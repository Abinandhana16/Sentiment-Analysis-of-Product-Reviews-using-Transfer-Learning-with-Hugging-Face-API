import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    print("Loading dataset...")
    # Using a small subset of the yelp review dataset as a proxy for product reviews
    # You can change 'yelp_polarity' to 'amazon_polarity' for larger, real product reviews
    dataset = load_dataset("yelp_polarity")
    
    # Take a small subset for rapid training/testing
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(200))

    # Load a pre-trained model and tokenizer
    # We use a tiny BERT for faster execution during prototyping
    model_name = "prajjwal1/bert-tiny" 
    print(f"Loading tokenizer and model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing data...")
    tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = small_eval_dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer API (Transfer Learning)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    print("Starting Transfer Learning...")
    trainer.train()

    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # Save the fine-tuned model
    save_directory = "./fine_tuned_sentiment_model"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")

if __name__ == "__main__":
    main()
