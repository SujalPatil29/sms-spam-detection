import streamlit as st
import pickle
import string
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Ensure the NLTK data is downloaded and stored in the right directory
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)  # Set NLTK data path
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)  # Tokenize text

    y = []
    for i in text:
        if i.isalnum():  
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load saved models with error handling
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found! Please check 'vectorizer.pkl' and 'model.pkl'.")

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("✉ Enter your message:")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("⚠ Please enter a valid message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.header("🚨 Spam" if result == 1 else "✅ Not Spam")
