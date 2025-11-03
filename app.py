import streamlit as st
from model import manual_testing, output_label  # your existing model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fake News Detector", layout="centered")

# --- CUSTOM BACKGROUND AND STYLING ---
st.markdown(
    """
    <style>
    /* Set background image */
    body {
        background-image: url("/background.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: #222;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Overlay for readability */
    .stApp {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 2rem;
        backdrop-filter: blur(3px);
    }

    /* Navigation bar */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-bottom: 30px;
        font-size: 18px;
        font-weight: 600;
    }
    button[kind="secondary"] {
        background-color: #fff !important;
        border: 1px solid #ccc !important;
        color: #333 !important;
    }
    button[kind="secondary"]:hover {
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
    }

    /* Titles */
    h1, h2, h3 {
        color: #1a1a1a;
    }

    /* Buttons */
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-size: 16px;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #e63e3e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SESSION STATE FOR NAVIGATION ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- NAVIGATION BAR ---
cols = st.columns([1, 1, 1])
if cols[0].button(" Home"):
    st.session_state.page = "Home"
if cols[1].button(" About Us"):
    st.session_state.page = "About"
if cols[2].button(" Detector"):
    st.session_state.page = "Detector"

# --- PAGE LOGIC ---

# HOME PAGE
if st.session_state.page == "Home":
    st.title("Fake News Detection System")
    st.markdown("""
    ### What is Fake News?
    Fake news refers to false or misleading information presented as legitimate news. 
    It often spreads quickly through social media and online platforms, influencing 
    public perception and decision-making.

    ### What is Real News?
    Real news is factual, verified, and published by credible journalists and media 
    organizations who adhere to professional ethics and verification standards.

    ### Why Detect Fake News?
    Detecting fake news is important for:
    -  Preventing misinformation
    -  Protecting public trust
    -  Encouraging responsible journalism

    ---
    """)
    st.markdown("### Ready to test some news?")
    if st.button(" Go to Fake News Detector"):
        st.session_state.page = "Detector"

# ABOUT US PAGE
elif st.session_state.page == "About":
    st.title(" About Us")
    st.markdown("""
    "Our **Fake News Detection System** is a project designed to help individuals and organizations distinguish between true and false news articles. Powered by machine learning, our tool analyzes text and provides you with accurate results on whether a news piece is *real* or *fake*."


    ###  Built With
    - Python & Streamlit    
    - Machine Learning (TF-IDF, Logistic Regression Algorithm)

    ###  Developed By
    Anjani Kumar Pokhrel \n
    2025  

    ---
    *Our goal is to help users identify misinformation and encourage critical thinking in the digital era.*
    """)

#  DETECTOR PAGE
elif st.session_state.page == "Detector":
    st.title(" Fake News Detection App")

    user_input = st.text_area("Enter a news article text:", height=250)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some news content.")
        else:
            try:
                prediction, confidence = manual_testing(user_input)
                label = output_label(prediction)

                # Select probability of predicted class
                confidence_score = confidence[prediction] * 100  

                # Show results
                if prediction == 0:
                    st.error(f"{label} (Confidence: {confidence_score:.2f}%)")
                else:
                    st.success(f"{label} (Confidence: {confidence_score:.2f}%)")

                # Confidence breakdown
                st.write("### Confidence Breakdown")
                st.progress(int(confidence_score))
                st.write(f"- Fake News Probability: {confidence[0]*100:.2f}%")
                st.write(f"- Genuine News Probability: {confidence[1]*100:.2f}%")

            except Exception as e:
                st.error(f"An error occurred: {e}")
