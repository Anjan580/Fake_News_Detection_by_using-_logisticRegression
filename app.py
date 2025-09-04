import streamlit as st
from model import manual_testing, output_label

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detection App")

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

            # Optional: Show both class probabilities
            st.write("### Confidence Breakdown")
            st.progress(int(confidence_score))
            st.write(f"- Fake News Probability: {confidence[0]*100:.2f}%")
            st.write(f"- Genuine News Probability: {confidence[1]*100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred: {e}")