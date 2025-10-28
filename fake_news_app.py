# fake_news_app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to check if it's **Real** or **Fake** using Machine Learning.")

# Step 1: Load and Prepare Dataset from Google Drive
@st.cache_data
def load_data():
    # Replace these with your actual Google Drive file IDs
    fake_url = "https://drive.google.com/uc?export=download&id=1lMQwKXtEyBkpmg3E3A7cQfvyN9PKNJo-"
    true_url = "https://drive.google.com/uc?export=download&id=1m-gIKg0il6Sk0MG51o3Ge7qtFLd7JfTT"

    df_fake = pd.read_csv(fake_url)
    df_true = pd.read_csv(true_url)

    df_fake["label"] = 0  # Fake
    df_true["label"] = 1  # Real

    df = pd.concat([df_fake, df_true])
    df = df.sample(frac=1).reset_index(drop=True)
    return df[["text", "label"]]

df = load_data()

# Step 2: Train Model (only once)
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    st.info(f"Model trained with accuracy: {acc:.2f}")

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
else:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

# Step 3: User Input and Prediction
user_input = st.text_area("Paste your news article here:", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        confidence = model.predict_proba(input_vector)[0][prediction]
        result = "✅ Real News" if prediction == 1 else "❌ Fake News"
        st.subheader("Result:")
        st.success(f"{result} (Confidence: {confidence:.2f})")