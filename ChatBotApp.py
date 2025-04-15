# chatbot_app.py

import streamlit as st
import nltk
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download NLTK resources (cached so it doesn’t re-run every time)
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')  # Optional if you use WordNetLemmatizer

setup_nltk()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data
model = load_model("chatbotmodel.h5")
intents = json.load(open("breastCancer.json"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Preprocessing functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get response
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        for i in intents_json["intents"]:
            if tag == i["tags"]:
                return i["responses"]
    return "Sorry, I do not understand."

# Streamlit UI
st.title("🩺 Breast Cancer Chatbot")
st.markdown("Ask me anything about breast cancer support, risk, symptoms, and more.")

user_input = st.text_input("You:", placeholder="Type your question here...")

if user_input:
    ints = predict_class(user_input)
    res = get_response(ints, intents)
    st.markdown(f"**Bot:** {res}")
