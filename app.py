import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

tokenizer=TreebankWordTokenizer()
ps=PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)

    # Keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]

    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    # Stemming
    tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)


# Load vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Email/SMS spam Classifier")

input_sms=st.text_input("Enter the message")

if st.button("Predict"):
    # 1. preprocess
    transform_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
