# NOTE: clean noisy transcripts with regex; use n-grams with tfidf; limit vocabulary; bag of words (in tfidf implicit)

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import numpy as np

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
plt.show()


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
plt.show()