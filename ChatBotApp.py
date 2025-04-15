# ChatBotApp.py

import streamlit as st
import json
import pickle
import numpy as np

# Lazy-load NLTK
import nltk
from nltk.stem import WordNetLemmatizer

@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
setup_nltk()

# Load model and assets
from tensorflow.keras.models import load_model

model = load_model("chatbotmodel.h5")
intents = json.load(open("breastCancer.json"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

lemmatizer = WordNetLemmatizer()

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

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        for i in intents_json["intents"]:
            if tag == i["tags"]:
                return i["responses"]
    return "Sorry, I do not understand."

# Streamlit App
st.set_page_config(page_title="Breast Cancer Chatbot", page_icon="💬")
st.title("🩺 Breast Cancer Chatbot")
st.markdown("Ask anything about breast cancer: risk, symptoms, prevention, and more.")

user_input = st.text_input("You:", placeholder="Ask your question here...")

if user_input:
    intents_result = predict_class(user_input)
    response = get_response(intents_result, intents)
    st.markdown(f"**Bot:** {response}")
