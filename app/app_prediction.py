import streamlit as st
import joblib
import numpy as np
import shap

from utils import clean_text

# -------------------------
# Load models
# -------------------------
model = joblib.load("../models/logistic_regression.pkl")
tfidf = joblib.load("../models/tfidf.pkl")

# -------------------------
# Title
# -------------------------
st.title("🧠 NLP Review Analyzer")

st.write("Enter a review to get predictions:")

# -------------------------
# Input
# -------------------------
user_input = st.text_area("✍️ Write your review here:")

# -------------------------
# Prediction
# -------------------------
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean text
        clean = clean_text(user_input)

        # Vectorize
        X = tfidf.transform([clean])

        # Predict rating
        pred = model.predict(X)[0]

        # Predict probabilities
        proba = model.predict_proba(X)[0]

        # -------------------------
        # Sentiment mapping
        # -------------------------
        if pred <= 2:
            sentiment = "Negative 😡"
        elif pred == 3:
            sentiment = "Neutral 😐"
        else:
            sentiment = "Positive 😊"

        # -------------------------
        # Display results
        # -------------------------
        st.subheader("📊 Results")

        st.write(f"⭐ Predicted Rating: **{pred}**")
        st.write(f"😊 Sentiment: **{sentiment}**")

        # -------------------------
        # Probabilities
        # -------------------------
        st.subheader("🔢 Prediction Probabilities")

        st.bar_chart(proba)

        # -------------------------
        # SHAP EXPLANATION
        # -------------------------
        st.subheader("🔍 Explanation (SHAP)")

        explainer = shap.LinearExplainer(model, tfidf.transform([clean]))

        shap_values = explainer([clean])

        st.write("Top contributing words:")

        shap.plots.text(shap_values[0])