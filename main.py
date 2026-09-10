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
#streamlit app
st.set_page_config(page_title="Sentiment Analysis", page_icon="●", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

.stApp { background: #0B0B0F; color: #F5F5F7; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding-top: 60px !important; max-width: 900px !important; }

.hero-label {
    font-family: 'Inter', sans-serif;
    font-size: 11px; font-weight: 500;
    color: #4A4A55; letter-spacing: 0.3em;
    text-transform: uppercase; margin-bottom: 24px;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 56px; font-weight: 500;
    line-height: 1.1; color: #F5F5F7;
    margin-bottom: 20px; letter-spacing: -0.02em;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 16px; line-height: 1.6;
    color: #8B8B95; max-width: 600px; margin-bottom: 48px;
}
.hero-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #4A4A55;
    letter-spacing: 0.05em; margin-bottom: 60px;
}
.hero-status .dot { color: #00D4A8; }

.stTextArea label {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important; font-weight: 600 !important;
    color: #00D4A8 !important; letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
}
.stTextArea textarea {
    background: #12121A !important;
    border: 1px solid #1F1F2B !important;
    color: #F5F5F7 !important; border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important; padding: 16px !important;
    min-height: 140px !important;
}
.stTextArea textarea:focus {
    border-color: #00D4A8 !important;
    box-shadow: 0 0 0 1px #00D4A8 !important;
}

.stButton > button {
    background: #00D4A8 !important; color: #0B0B0F !important;
    border: none !important; border-radius: 4px !important;
    padding: 14px 40px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important; font-weight: 600 !important;
    letter-spacing: 0.2em !important; text-transform: uppercase !important;
    margin-top: 20px !important; transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #F5F5F7 !important; transform: translateY(-1px) !important;
}

.result-card {
    margin-top: 48px; padding: 48px;
    background: #12121A; border: 1px solid #1F1F2B;
    border-radius: 4px; animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 112px; font-weight: 500;
    line-height: 1; letter-spacing: -0.04em; margin-bottom: 20px;
}
.result-big.positive { color: #00D4A8; }
.result-big.negative { color: #FF4D6D; }
.result-verdict {
    font-family: 'Fraunces', serif;
    font-size: 26px; font-weight: 500;
    color: #F5F5F7; line-height: 1.3; margin-bottom: 12px;
}
.result-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #4A4A55;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.confidence-bar {
    height: 4px; background: #2A2A3A; border-radius: 2px;
    overflow: hidden; margin-top: 24px;
}
.confidence-fill { height: 100%; transition: width 0.6s ease; }

.empty-state {
    text-align: center; padding: 60px 20px;
    color: #4A4A55; font-style: italic;
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-label">Sentiment Analysis · Deep Learning</div>
<div class="hero-title">Every review carries a signal.<br/>The model reads it in milliseconds.</div>
<div class="hero-sub">A SimpleRNN trained on 25,000 IMDB movie reviews. Paste any review below and see how the network scores its sentiment.</div>
<div class="hero-status"><span class="dot">●</span> &nbsp;Model live &nbsp;·&nbsp; SimpleRNN · 128 units &nbsp;·&nbsp; Vocab 10K</div>
""", unsafe_allow_html=True)

# Input
user_input = st.text_area("Movie review", placeholder="Type or paste a movie review here...")

if st.button("Classify sentiment"):
    if user_input.strip():
        prediction = predict_review(user_input)
        sentiment, score = prediction
        
        is_positive = sentiment == "Positive"
        confidence = score * 100 if is_positive else (1 - score) * 100
        css_class = "positive" if is_positive else "negative"
        verdict = "This review leans positive." if is_positive else "This review leans negative."
        bar_color = "#00D4A8" if is_positive else "#FF4D6D"
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-big {css_class}">{confidence:.0f}%</div>
            <div class="result-verdict">{verdict}</div>
            <div class="result-meta">Raw model score: {score:.4f} &nbsp;·&nbsp; Threshold: 0.5</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%; background: {bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Enter a review first.")
else:
    st.markdown('<div class="empty-state">Enter a review above and click Classify to see the prediction.</div>', unsafe_allow_html=True)