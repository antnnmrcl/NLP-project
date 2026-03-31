import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def semantic_search(query, query_emb, embeddings, df, top_k=5):
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_idx = scores.argsort()[-top_k:][::-1]
    return df.iloc[top_idx]