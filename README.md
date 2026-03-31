# NLP Project: Insurance Reviews Analysis

Authors : Isabella LEWIS & Antonin MARCEL

## 📝 Project Overview

This project analyzes **35 Excel datasets** of French insurance reviews, containing both the original French text and English translations.  
The goal is to build a **full NLP pipeline** including data cleaning, exploratory analysis, supervised learning, embeddings, topic modeling, and interactive Streamlit apps.

The workflow is divided into three phases:

1. **Data Cleaning & EDA**  
2. **Supervised NLP Modeling & Prediction App**  
3. **Unsupervised Analysis & Insurer Dashboard App**

---

## 📂 Project Structure


---

## ⚙️ Python Environment & Libraries

- **Data handling & visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`  
- **NLP & preprocessing:** `nltk`, `spacy`, `pyspellchecker`, `re`  
- **Supervised ML:** `scikit-learn`, `joblib`  
- **Deep learning & embeddings:** `sentence-transformers`, `transformers` (optional for summarization)  
- **Topic modeling:** `BERTopic`, `gensim` (LDA)  
- **Interactive apps:** `streamlit`  
- **Explainability:** `shap`  

> Notes:
> - `pyspellchecker` is used for optional spelling correction.  
> - `spacy` handles tokenization and lemmatization (French & English).  
> - `sentence-transformers` provides semantic embeddings for search and clustering.

---

## 🧹 Phase 1 — Data Cleaning & EDA

1. **Data Loading & Merging**  
   - Loaded all 35 Excel files (`avis_#.xlsx`)  
   - Merged into a single dataset  
   - Checked column consistency and encoding  

2. **Data Cleaning**  
   - Removed rows with missing `note`  
   - Converted dates to `datetime`  
   - Removed duplicates  
   - Text preprocessing:
     - Lowercase  
     - Punctuation removal  
     - Stopword removal (French & English)  
     - Lemmatization  
     - Optional spelling correction  

3. **Feature Engineering**  
   - `clean_text` → preprocessed review  
   - `text_length` → number of words  
   - `tokens` → token list  

4. **Exploratory Data Analysis**  
   - Distribution of ratings  
   - Reviews per insurer & product  
   - Time trends of reviews  
   - Word frequency and n-grams  
   - Word clouds  

---

## 🤖 Phase 2 — Supervised NLP Models

1. **Modeling Tasks**  
   - Rating prediction (`note`)  
   - Sentiment classification  
   - Optional topic classification (zero-shot)

2. **Models Used**  
   - Classical ML:
     - TF-IDF + Logistic Regression ✅
     - Random Forest, SVM (optional)
   - Deep Learning:
     - LSTM, CNN (embedding layer)
   - Pretrained Transformers:
     - BERT, CamemBERT, DistilBERT (French)

3. **Evaluation**  
   - Train/test split & cross-validation  
   - Metrics: accuracy, F1-score, confusion matrix  
   - Error analysis on mispredictions  

4. **Explainability**  
   - **SHAP LinearExplainer** for TF-IDF + Logistic Regression  
   - Displays **top contributing words**  
   - Highlighted words in the processed text for user understanding  

---

## 🖥 Phase 2 — Streamlit Prediction App

- **Input:** User review text  
- **Output:**
  - Predicted star rating  
  - Sentiment (Positive / Neutral / Negative)  
  - Top contributing words (SHAP explanation)  
  - Highlighted important words in text  

- **Key Features:**
  - Preprocessing applied automatically  
  - Probabilities shown as bar chart  
  - SHAP explanations adapted for TF-IDF  

- **Run app:**  
```bash
cd app
streamlit run app_prediction.py
```