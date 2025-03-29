import streamlit as st
import pickle
import string
import nltk

from nltk.tokenize import word_tokenize  # Fix tokenization
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Download required NLTK data (ensure these are available)
nltk.download('punkt')
nltk.download('stopwords')


def transform_text(text):
    text = text.lower()

    # Ensure tokenization works correctly
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)

    text = y[:]  # Copy filtered words
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]  # Copy again after removing stopwords
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    return " ".join(y)


# Load saved models with error handling
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found! Please check 'vectorizer.pkl' and 'model.pkl'.")

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a valid message.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        st.header("Spam" if result == 1 else "Not Spam")
