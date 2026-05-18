# NOTE: clean noisy transcripts with regex; use n-grams with tfidf; limit vocabulary; bag of words (in tfidf implicit)
# plot palavras relevantes? - Tirar verbos, usar apenas nomes commons

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import numpy as np
from collections import Counter
import re


folder = 'Project_Features'


# TFIDF
documents = []
doc_names = []
for file in os.listdir(folder):
    if file.endswith('_speech.pkl'):
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        full_text = " ".join(df['transcript'].astype(str))
        documents.append(full_text)
        doc_names.append(file.replace('_speech.pkl', ''))

nltk.download('stopwords')
stop_words_pt = stopwords.words('portuguese')
vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
tfidf = vectorizer.fit_transform(documents)
similarity_matrix = cosine_similarity(tfidf)

plt.figure(figsize=(10, 8))
sns.heatmap(
    similarity_matrix,
    cmap="Blues",
    annot=False,              
    xticklabels=False,        
    yticklabels=doc_names    
)
plt.title("Debate Similarity (Cosine TF-IDF)")
plt.tight_layout()
plt.savefig("plots_v5/Debate_Similarity_TFIDF")
# plt.show()


# EMBEDDINGS
embeddings = []
doc_names = []
for file in os.listdir(folder):
    if file.endswith('_speech.pkl'):
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        emb_matrix = np.vstack(df['text_embedding'].values)
        mean_emb = emb_matrix.mean(axis=0) # average embbedings - meanpool
        embeddings.append(mean_emb)
        doc_names.append(file.replace('_speech.pkl', ''))
embeddings = np.array(embeddings)
embeddings = embeddings - embeddings.mean(axis=0) # Recenter embeddings (to get more meaningful cosine similarities)
embedding_similarity = cosine_similarity(embeddings)

plt.figure(figsize=(10, 8))
sns.heatmap(
    embedding_similarity,
    cmap="Blues",         
    annot=False,
    xticklabels=False,
    yticklabels=doc_names
)
plt.title("Debate Similarity (Embeddings)")
plt.tight_layout()
plt.savefig("plots_v5/Debate_Similarity_Embeddings")
# plt.show()


for file in os.listdir(folder):
    if file.endswith('_speech.pkl'):
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        full_text = " ".join(df['transcript'].astype(str)).lower()
        full_text = re.sub(r'[^a-zà-ÿ\s]', ' ', full_text)
        words = full_text.split()
        words = [
            w for w in words
            if w not in stop_words_pt and len(w) > 2
        ]
        word_counts = Counter(words)
        top_words = word_counts.most_common(20)
        if len(top_words) == 0:
            continue
        words_plot = [w[0] for w in top_words]
        counts_plot = [w[1] for w in top_words]
        plt.figure(figsize=(10, 6))
        plt.barh(words_plot[::-1], counts_plot[::-1])
        debate_name = file.replace('_speech.pkl', '')
        plt.title(f"Most Common Words | {debate_name}")
        plt.xlabel("Frequency")
        plt.ylabel("Words")
        plt.tight_layout()
        plt.savefig(
            f"plots_v5/per_debate/{debate_name}_most_common_words.png",
            bbox_inches='tight'
        )
        plt.close()