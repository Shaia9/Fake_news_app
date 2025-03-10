import pandas as pd
import streamlit as st
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from train_model import df_fake

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

# load for visualization
df_fake = pd.read_csv("fake.csv")
df_true = pd.read_csv("true.csv")

# Labels for visualization
df_fake["label"] = "Fake News"
df_true["label"] = "True News"
df = pd.concat([df_fake, df_true])

# Sidebar for Data
st.sidebar.header("Explore Dataset")
if st.sidebar.button("Show Class Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="label", palette="coolwarm", ax=ax)
    st.pyplot(fig)