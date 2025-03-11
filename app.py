import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import df_fake, y_test
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Manual install for Streamlit
import os
os.system("pip install -r requirements.txt")

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
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Labels for visualization
df_fake["label"] = "Fake News"
df_true["label"] = "True News"
df = pd.concat([df_fake, df_true])

# Sidebar for Data
st.sidebar.header("Explore Dataset and Performance")

# Histogram: Fake v. Real Distribution
if st.sidebar.checkbox("Show Data Distribution"):
    st.subheader("Fake v. Real Distribution")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="label", palette="coolwarm", ax=ax)
    plt.xlabel("News Type")
    plt.ylabel("Count")
    plt.title("Distribution of Fake vs. Real News")
    st.pyplot(fig)

# Confusion Matrix
if st.sidebar.checkbox("Show Performance"):
    st.sidebar.checkbox("Model Performance")
    st.subheader("Confusion Matrix")
    y_test = pd.read_csv("y_test.csv").squeeze()
    y_pred = pd.read_csv("y_pred.csv").squeeze()
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# Word Cloud: Most common words in Fake News
if st.sidebar.checkbox("Most Common Words in Fake News"):
    fake_text = " ".join(df_fake["text"].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(fake_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.title("Most Common Words in Fake News")
    st.pyplot(fig)