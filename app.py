import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # import vectorizer type based on what you used

# Title and Description
st.title("Spam Detection Model")
st.write("This application predicts if a given email is spam or not using an NLP model trained on the Enron dataset.")



# Load Model and Vectorizer
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:  # assuming vectorizer was saved as 'vectorizer.pkl'
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model()

# Text Input for Prediction
st.write("### Enter an Email Text to Classify")
email_text = st.text_area("Paste the email content here")

# Process and Predict
if email_text:
    # Convert input text to the vector format expected by the model
    email_vector = vectorizer.transform([email_text])
    
    # Make Prediction
    prediction = model.predict(email_vector)[0]  # 0 for single prediction

    # Display Result
    result = "Spam" if prediction == 1 else "Not Spam"
    st.write("Prediction:", result)
