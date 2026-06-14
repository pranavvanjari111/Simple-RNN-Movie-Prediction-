import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎬 Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.main-title {
    text-align: center;
    color: white;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-bottom: 30px;
}

.result-positive {
    background: linear-gradient(135deg,#16a34a,#22c55e);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.result-negative {
    background: linear-gradient(135deg,#dc2626,#ef4444);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    background-color: rgba(255,255,255,0.08);
    margin-top: 15px;
}

.footer {
    text-align:center;
    color:#94a3b8;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model("simple_rnn_imdb.h5")

# =========================
# HELPER FUNCTIONS
# =========================

def decode_review(encoded_review):
    return " ".join(
        [reverse_word_index.get(i - 3, "?") for i in encoded_review]
    )


def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


def predict_sentiment(review):
    processed = preprocess_text(review)

    prediction = model.predict(
        processed,
        verbose=0
    )

    sentiment = (
        "Positive"
        if prediction[0][0] > 0.5
        else "Negative"
    )

    return sentiment, float(prediction[0][0])

# =========================
# HEADER
# =========================

st.markdown(
    '<div class="main-title">🎬 Movie Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Analyze IMDb Movie Reviews using Deep Learning (Simple RNN)</div>',
    unsafe_allow_html=True
)

st.divider()

# =========================
# INPUT SECTION
# =========================

user_input = st.text_area(
    "📝 Enter Movie Review",
    height=180,
    placeholder="Example: This movie was amazing. The acting and storyline were fantastic..."
)

predict_button = st.button(
    "🔍 Analyze Review",
    use_container_width=True
)

# =========================
# PREDICTION
# =========================

if predict_button:

    if not user_input.strip():
        st.warning("⚠️ Please enter a movie review.")
    else:

        with st.spinner("Analyzing Review..."):

            sentiment, prediction = predict_sentiment(
                user_input
            )

        st.divider()

        if sentiment == "Positive":

            st.markdown(
                """
                <div class="result-positive">
                😊 POSITIVE REVIEW
                </div>
                """,
                unsafe_allow_html=True
            )

            confidence = prediction

        else:

            st.markdown(
                """
                <div class="result-negative">
                😔 NEGATIVE REVIEW
                </div>
                """,
                unsafe_allow_html=True
            )

            confidence = 1 - prediction

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction Score",
                f"{prediction:.4f}"
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.write("### 📊 Confidence Level")
        st.progress(float(confidence))

        st.write("### 📝 Review")
        st.info(user_input)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("ℹ️ About")

    st.write("""
    This application uses a **Simple RNN**
    trained on the IMDb Movie Review Dataset.

    The model predicts whether a movie review
    is:

    ✅ Positive

    ❌ Negative
    """)

    st.write("---")

    st.write("**Tech Stack**")
    st.write("• TensorFlow")
    st.write("• Keras")
    st.write("• Streamlit")
    st.write("• Simple RNN")

# =========================
# FOOTER
# =========================

st.markdown(
    """
    <div class="footer">
    Built with ❤️ using TensorFlow & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)