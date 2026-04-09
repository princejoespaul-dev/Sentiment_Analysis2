import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎬 Movie Review Sentiment Analyzer")
st.caption("Analyze movie reviews using AI")

# Input
review = st.text_area(
    "✍️ Enter your movie review:",
    height=150,
    placeholder="Example: This movie was absolutely amazing with great acting!"
)

# Button
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("⏳ Analyzing your review..."):
            try:
                response = requests.post(
                    "https://sentiment-analysis2-b9co.onrender.com/predict",
                    json={"reviews": review},   # ✅ FIXED KEY
                    timeout=10
                )

                # Debug (optional)
                # st.text(response.text)

                if response.status_code == 200:
                    result = response.json()

                    sentiment = result.get('prediction', 'Unknown')
                    confidence = result.get('confidence', None)

                    # Display result with style
                    if "pos" in sentiment.lower():
                        st.success(f"😊 Positive Sentiment: **{sentiment}**")
                    elif "neg" in sentiment.lower():
                        st.error(f"😡 Negative Sentiment: **{sentiment}**")
                    else:
                        st.info(f"🤔 Sentiment: **{sentiment}**")

                    # Confidence (optional)
                    if confidence:
                        st.metric("Confidence Score", f"{confidence:.2f}")

                else:
                    st.error(f"🚨 API Error: {response.status_code}")
                    st.text(response.text)

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Server might be slow (Render free tier 😅).")

            except requests.exceptions.ConnectionError:
                st.error("🌐 Cannot connect to API. Check if backend is running.")

            except Exception as e:
                st.error("⚠️ Something went wrong!")
                st.text(str(e))
