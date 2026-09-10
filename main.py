import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#convertinf indexes
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

#loading model
model = load_model("simpleRNN_imdb_relu.h5")

#helper functions
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#prediction function
def predict_review(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

import streamlit as st
#streamlit app 

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below, and the model will predict whether the sentiment is positive or negative.")

user_input = st.text_area("Enter a movie review:")

if st.button("Classify"):

    prediction = predict_review(user_input)
    sentiment, score = prediction
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Score: {score:.4f}")

else:
    st.write("Please enter a review and click 'Classify' to see the prediction.")