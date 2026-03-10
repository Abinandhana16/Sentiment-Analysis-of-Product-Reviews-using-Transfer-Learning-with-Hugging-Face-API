import gradio as gr
from transformers import pipeline
import os

# Check if model exists locally, otherwise use a default HF hub model
MODEL_DIR = "./fine_tuned_sentiment_model"

if os.path.exists(MODEL_DIR):
    print(f"Loading local fine-tuned model from {MODEL_DIR}")
    sentiment_pipeline = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)
else:
    print("Local model not found! Loading default distilbert-base-uncased-finetuned-sst-2-english.")
    sentiment_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(review_text):
    if not review_text.strip():
        return "Please enter a product review."
    
    result = sentiment_pipeline(review_text)[0]
    
    label = result['label']
    score = result['score']
    
    # Format the output nicely
    if label in ["LABEL_1", "POSITIVE"]:
        sentiment = "Positive 🎉"
    elif label in ["LABEL_0", "NEGATIVE"]:
        sentiment = "Negative 😞"
    else:
        sentiment = label
        
    return f"Sentiment: {sentiment}\nConfidence: {score:.2%}"

# Create a beautiful Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")) as demo:
    gr.Markdown(
        """
        # 🛍️ Product Review Sentiment Analysis
        Analyze the sentiment of your customer reviews using Transfer Learning and Hugging Face APIs.
        """
    )
    
    with gr.Row():
        with gr.Column():
            review_input = gr.Textbox(
                label="Enter Product Review",
                placeholder="e.g., 'This product is absolutely amazing! I highly recommend it.'",
                lines=5
            )
            analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
            
        with gr.Column():
            output_display = gr.Textbox(label="Analysis Result", lines=6, interactive=False)
            
    # Example reviews
    gr.Examples(
        examples=[
            ["The battery life on this laptop is phenomenal, easily lasts all day!"],
            ["Terrible experience. The item broke after just two uses. Do not buy."],
            ["It's okay for the price, but the build quality feels a bit cheap."],
            ["Best purchase I've made all year. Fast shipping and excellent customer service."]
        ],
        inputs=review_input
    )
    
    analyze_btn.click(fn=analyze_sentiment, inputs=review_input, outputs=output_display)

if __name__ == "__main__":
    demo.launch()
