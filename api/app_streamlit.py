import streamlit as st
import requests
import plotly.express as px

# -----------------------------
# Configuration
# -----------------------------
API_URL = "http://127.0.0.1:8000"  # Adjust if your API runs elsewhere

st.set_page_config(
    page_title="Tweet Sentiment Analysis",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("🧠 Tweet Sentiment Analysis")
st.markdown("""
Predict the sentiment of your tweets with confidence scores and visualize explanations using LIME.
""")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
- Enter a tweet (max 280 characters)
- Click **Predict Sentiment** for a quick sentiment
- Click **Explain (LIME)** for detailed word-level explanation
- Green = Positive, Red = Negative
""")
    st.image("https://docs.streamlit.io/logo.svg", width=200)
    st.write("API status will be tested when you predict.")

# -----------------------------
# Input Section
# -----------------------------
tweet_text = st.text_area(
    "Enter your tweet (max 280 characters):",
    max_chars=280,
    placeholder="I love this product! It's amazing..."
)

# Validation
text_valid = bool(tweet_text.strip())
if text_valid:
    st.success(f"✅ Text valid ({len(tweet_text)}/280 characters)")
else:
    st.warning("⛔ Please enter a valid text.")

# -----------------------------
# Action Buttons
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🎯 Predict Sentiment", disabled=not text_valid)
with col2:
    explain_btn = st.button("🔍 Explain with LIME (30-60s)", disabled=not text_valid)

# -----------------------------
# API Calls
# -----------------------------
def call_predict_api(text):
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API /predict error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"API /predict error: {e}")
        return None

def call_explain_api(text):
    try:
        response = requests.post(f"{API_URL}/explain", json={"text": text}, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API /explain error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"API /explain error: {e}")
        return None

# -----------------------------
# Display Predict Results
# -----------------------------
if predict_btn:
    data = call_predict_api(tweet_text)
    if data:
        sentiment = data["sentiment"].capitalize()
        confidence = data["confidence"]
        prob_pos = data["probability_positive"]
        prob_neg = data["probability_negative"]

        st.subheader("Prediction Result")
        if sentiment.lower() == "positif" or sentiment.lower() == "positive":
            st.success(f"😊 POSITIVE ({confidence:.1%})")
        else:
            st.error(f"😞 NEGATIVE ({confidence:.1%})")

        # Plotly bar chart
        fig = px.bar(
            x=["Negative", "Positive"],
            y=[prob_neg, prob_pos],
            labels={'x':'Sentiment', 'y':'Probability'},
            color=["red", "green"],
            color_discrete_map={"Negative": "red", "Positive": "green"},
            text=[f"{prob_neg:.1%}", f"{prob_pos:.1%}"]
        )
        fig.update_layout(title="Probability Scores", yaxis=dict(range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Display LIME Explanation
# -----------------------------
if explain_btn:
    explanation_data = call_explain_api(tweet_text)
    if explanation_data:
        st.subheader("🔍 LIME Explanation")
        st.components.v1.html(
            explanation_data["html_explanation"],
            height=400,
            scrolling=True
        )
        with st.expander("📊 Detailed Explanation"):
            st.write(explanation_data["explanation"])
