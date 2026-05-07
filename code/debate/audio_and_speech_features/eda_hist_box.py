# NOTE: Bag of words, pos, remove stopwords, box plots, histograms, clustering or manual-label by speaker, tsne/pca/umap,
        # outlier detection, normalization like z-score and word_count/duration, missing data, alinhar timespamps, 
        # correlations, simnilarities (e.g. pairwise heatmaps). t-test?; average/sum per debate  then compare
# NOTE: Order by time; ganttstyle; Begining debate=end??; alinhar debates; time-series
# NOTE: Could be interesting Clustering (by speaker for example), but i only know about K-Means DBSCAN :/
# NOTE: PCA/t-sne for embeddings?
# NOTE: Focus 1 debate?; debates one aparece candidato X; do multi debate already? Relations betwwen features live Visual+Speech = Emotion or speaker X
# TODO: PCA 3 Candidates Speaking; PCA for keypoints; cluster by speeaker; cluester by topics; cluster by similar debates; Correlations inter-feature
# Do not only per numeric feature on 1 debate but also across all debates - Done
# What does each feature mean - interpret
# plavvras mais relevantes por debate (eda_v5) - plot
# TODO: 1 debate em 1 telejornal (podemos ignorar - mencionar isso); janeiro de 2025 para jan 2026; <br> por espaco (preprocessamento)

import pandas as pd
import os
import matplotlib.pyplot as plt

folder = 'Project_Features'
file = 'Ventura_vs_Marques_Mendes_November_25_audio.pkl'
data = pd.read_pickle(os.path.join(folder, file))
output_dir = 'plots_v3/ventura-mm'
os.makedirs(output_dir, exist_ok=True)

# Select scalar numeric columns 
numeric_cols = []
for col in data.columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        if not data[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            numeric_cols.append(col)
print(numeric_cols)

# EDA - Do one Histogram and Boxplot per numeric speech feature 
i = 0
while i < len(numeric_cols):
    col = numeric_cols[i]
    values = data[col].dropna()
    if len(values) == 0 or values.nunique() <= 1:
        i += 1
        continue
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(values, bins=30)
    axes[0].set_title("Histogram")
    axes[1].boxplot(values, vert=True)
    axes[1].set_title("Boxplot")
    fig.suptitle(f"{file} | {col}")
    plt.tight_layout()
    safe_col = col.replace(" ", "_").replace("/", "_")
    filename = f"{file}_{safe_col}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Saved: {filepath}")
    i += 1

# ------------------------------------------------------------------------------------------------------------------------------

# EDA for ALL Debates
folder = 'Project_Features'
output_dir = 'plots_v3/all_debates'
os.makedirs(output_dir, exist_ok=True)

# Load all debates
all_data = []
for file in os.listdir(folder):
    if file.endswith('_audio.pkl'):
        filepath = os.path.join(folder, file)
        try:
            data = pd.read_pickle(filepath)
            data['debate_file'] = file
            all_data.append(data)
            print(f"Loaded: {file} | shape={data.shape}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

data = pd.concat(all_data, ignore_index=True)
print("\nCombined shape:", data.shape)

# Organize mumeric columns
numeric_cols = []
for col in data.columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        if not data[col].apply(
            lambda x: isinstance(x, (list, tuple))
        ).any():
            numeric_cols.append(col)
print("\nNumeric columns:")
print(numeric_cols)

# Plots of Histograms and Boxplots
for col in numeric_cols:
    values = data[col].dropna()
    if len(values) == 0 or values.nunique() <= 1:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(values, bins=30)
    axes[0].set_title("Histogram")
    axes[0].set_xlabel(col)
    axes[1].boxplot(values, vert=True)
    axes[1].set_title("Boxplot")
    fig.suptitle(f"ALL DEBATES | {col}")
    plt.tight_layout()
    safe_col = (
        col.replace(" ", "_")
           .replace("/", "_")
           .replace("\\", "_")
    )
    filename = f"ALL_DEBATES_{safe_col}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")

print("\nDone.")