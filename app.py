import streamlit as st
import pickle
import string
import os
import nltk
nltk.download('punkt_tab')

# Set up NLTK data directory manually to prevent lookup errors
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK datasets properly
try:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)  # Added for better text processing
except Exception as e:
    st.error(f"🚨 NLTK data download failed: {e}")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure stopwords are loaded properly
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    st.warning("Downloading stopwords again...")
    nltk.download('stopwords', download_dir=nltk_data_path)
    stop_words = set(stopwords.words('english'))

# Initialize stemmer
ps = PorterStemmer()


# Function to transform input text
def transform_text(text):
    text = text.lower()

    try:
        tokens = word_tokenize(text)  # Ensure punkt tokenizer is working
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)
        tokens = word_tokenize(text)

    # Keep only alphanumeric tokens & remove stopwords
    filtered_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]

    return " ".join(filtered_tokens)


# Load saved models with error handling
tfidf, model = None, None
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 Model files not found! Please check 'vectorizer.pkl' and 'model.pkl'.")

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("✉ Enter your message:")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("⚠ Please enter a valid message.")
    elif tfidf is None or model is None:
        st.error("🚨 Model files are missing. Cannot proceed with prediction.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        st.header("🚨 Spam" if result == 1 else "✅ Not Spam")
