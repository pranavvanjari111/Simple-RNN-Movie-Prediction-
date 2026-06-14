import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the word index
word_index= imdb.get_word_index()
reverse_word_index = {value : key for key, value in word_index.items()}

# Load the pre-trained model with relu activation
model = load_model('simple_rnn_imdb.h5')


# helper functions

def decode_review(encded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encded_review])


# preprocessing the user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## Create the Prediction function

def predict_sentiment(review):
    processed = preprocess_text(review)
    prediction = model.predict(processed)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]


## streamlit App
import streamlit as st
st.title("Sentiment Analysis on the Moive Reviews")
st.write("Enter the movie review to classify is it Positive or Negative")

user_input = st.text_input("Enter movie review")

if st.button("Classify Movie Review"):


    #make the prediction
    sentiment, prediction = predict_sentiment(user_input)

    # display the results
    st.write("Movie Review:", user_input)
    st.write("Movie Review:", sentiment)
    st.write("Movie Review:", prediction)
else:
    st.write("Please enter a movie review")

