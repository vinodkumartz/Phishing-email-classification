import streamlit as st
import joblib
from transformers import AutoTokenizer, AutoModel
import torch

# Title and Description
st.title("Spam Detection Model")
st.write("This application predicts if a given email is spam or not using an NLP model trained on the Enron dataset.")

# Load Model and BERT Tokenizer/Model
def load_model():
    # Load the pre-trained ML model
    with open("model.pkl", "rb") as model_file:
        model = joblib.load(model_file)

    # Load the BERT tokenizer and model for embedding generation
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    
    return model, tokenizer, bert_model

model, tokenizer, bert_model = load_model()

# Text Input for Prediction
st.write("### Enter an Email Text to Classify")
email_text = st.text_area("Paste the email content here")

# Function to generate BERT embeddings
def get_bert_embedding(text, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Get the embeddings by averaging the last hidden layer
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Process and Predict
if email_text:
    # Convert input text to BERT embeddings
    email_vector = get_bert_embedding(email_text, tokenizer, bert_model)

    # Make Prediction
    prediction = model.predict(email_vector.numpy())[0]  # Ensure model expects 2D input shape

    # Display Result
    result = "Spam" if prediction == 1 else "Not Spam"
    st.write("Prediction:", result)

