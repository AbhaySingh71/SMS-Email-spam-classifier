import os
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Ensure the required NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

# Initialize Porter Stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def transform_text(text):
    # Lowercase
    text = text.lower()
    # Tokenization
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    # Stemming
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Load TF-IDF vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    
    # Check if the vectorizer is fitted by checking for 'idf_' attribute
    if not hasattr(tfidf, 'idf_'):
        raise ValueError("TF-IDF vectorizer is not fitted.")
except FileNotFoundError:
    st.error("Required files (vectorizer.pkl, model.pkl) are missing!")
except ValueError as e:
    st.error(f"Error: {e}")

# Streamlit app
st.title("📩 Email/SMS Spam Classifier")
st.write("This app helps classify messages as **Spam** or **Not Spam** using a machine learning model.")

# Input from user
input_sms = st.text_area("Enter the message to classify:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        # Vectorize the input
        vector_input = tfidf.transform([transformed_sms])
        # Predict using the model
        result = model.predict(vector_input)[0]
        # Display the result
        if result == 1:
            st.header("🚨 Spam!")
        else:
            st.header("✅ Not Spam!")
