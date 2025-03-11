import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from typing import BinaryIO

# Load data
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Assign: fake = 0, 1 = true
df_fake["label"] = 0
df_true["label"] = 1

# Combine data
df = pd.concat([df_fake, df_true])

# Select columns
df = df[["text", "label"]]

# Drop empty values
df = df.dropna()

# Split data
X = df["text"]
y = df["label"]

# Convert into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_transformed = vectorizer.fit_transform(X)

# Train Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Test Performance
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

# Save report to file
with open("model_evaluation.txt", "w") as f:
    f.write(report)

# Save predictions for Streamlit
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)
pd.DataFrame(y_pred).to_csv("y_pred.csv", index=False)

# Generate Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save report for Streamlit
report_df.to_csv("classification_report.csv")

# Print results in terminal
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
