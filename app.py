import streamlit as st
import pickle
import string
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Ensure NLTK downloads are available
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Ensure stopwords work
stop_words = set(stopwords.words('english'))


def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)  # Tokenization

    y = [i for i in text if i.isalnum()]  # Remove special characters

    # Remove stopwords & punctuation
    y = [i for i in y if i not in stop_words and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)


# Cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except FileNotFoundError:
        st.error("Model files not found! Ensure 'vectorizer.pkl' and 'model.pkl' exist.")
        return None, None


tfidf, model = load_models()

st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("✉ Enter your message:")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("⚠ Please enter a valid message.")
    elif tfidf and model:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.header("🚨 Spam" if result == 1 else "✅ Not Spam")
