import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("../data/processed/avis.csv")

# Load embeddings (precomputed)
embeddings = np.load("../models/embeddings.npy")

# Load sentence-transformer model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# App UI
# -------------------------
st.title("📊 Insurer Analytics Dashboard")

# Sidebar filters
insurer_filter = st.sidebar.multiselect(
    "Select insurer(s):", options=df['assureur'].unique(), default=df['assureur'].unique()
)

product_filter = st.sidebar.multiselect(
    "Select product(s):", options=df['produit'].unique(), default=df['produit'].unique()
)

df_filtered = df[
    (df['assureur'].isin(insurer_filter)) &
    (df['produit'].isin(product_filter))
]

st.subheader(f"Filtered dataset: {len(df_filtered)} reviews")

# -------------------------
# KPI metrics
# -------------------------
st.metric("Average Rating", round(df_filtered['note'].mean(), 2))
st.metric("Positive Reviews (%)", round((df_filtered['note']>=4).mean()*100, 2))
st.metric("Negative Reviews (%)", round((df_filtered['note']<=2).mean()*100, 2))

# -------------------------
# Rating distribution
# -------------------------
st.subheader("Rating Distribution")
fig = px.histogram(df_filtered, x='note', nbins=5, color='note', text_auto=True)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Topic distribution (optional: if you have topic labels)
# -------------------------
if 'topic' in df_filtered.columns:
    st.subheader("Topic Distribution")
    topic_counts = df_filtered['topic'].value_counts().reset_index()
    topic_counts.columns = ['topic', 'count']
    fig2 = px.bar(topic_counts, x='topic', y='count', text='count')
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Keyword search
# -------------------------
st.subheader("🔍 Keyword Search")
keyword = st.text_input("Search reviews by keyword:")

if keyword:
    keyword = keyword.lower()
    results = df_filtered[df_filtered['clean_text'].str.contains(keyword)]
    st.write(f"Found {len(results)} reviews:")
    st.dataframe(results[['assureur','produit','note','avis']].head(20))

# -------------------------
# Semantic search
# -------------------------
st.subheader("🧠 Semantic Search")
query = st.text_input("Search reviews semantically:")

if query:
    query_clean = clean_text(query)
    query_emb = embed_model.encode([query_clean])
    scores = cosine_similarity(query_emb, embeddings[df_filtered.index])[0]
    top_idx = scores.argsort()[-5:][::-1]
    st.write("Top matching reviews:")
    for i in top_idx:
        st.write(f"⭐ {df_filtered.iloc[i]['note']} | {df_filtered.iloc[i]['avis']}")

# -------------------------
# NLP Summaries (optional)
# -------------------------
st.subheader("📝 Summary of Reviews")
if st.button("Generate Summary"):
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        text = " ".join(df_filtered['avis'].head(50))
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        st.write(summary)
    except Exception as e:
        st.write("Summarization requires transformers + torch:", e)