import streamlit as st
from model import manual_testing, output_label

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter a news article text:", height=250)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        prediction = manual_testing(user_input)
        label = output_label(prediction)
        
        if prediction == 0:
            st.error(f"❌ {label}")
        else:
            st.success(f"✅ {label}")
