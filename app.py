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

def transform_text(text):
    text=text.lower()
    text=tokenizer.tokenize(text)

    y=[]
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
