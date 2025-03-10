import streamlit as st
import pickle

from train_model import vectorizer

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Fake News Detector")
st.write("Enter a news article to check if it's real or fake.")

user_input = st.text_area("Your News Article:", height=200)

if st.button("Check Credibility"):
    if user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text[0])
        label = "TRUE NEWS" if prediction == 1 else "FAKE NEWS"
        st.success(f"Predicted category: **{label}**")