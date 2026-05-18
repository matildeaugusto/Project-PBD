import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =====================================================
# CONFIG
# =====================================================
folder = "Project_Features"
target = "Marques_Mendes_vs_Gouveia_Melo_December_21"
output_folder = "part2_v04"

# =====================================================
# LOAD AUDIO FILE
# =====================================================
audio_df = None
for file in os.listdir(folder):
    if file.endswith("_audio.pkl") and target in file:
        audio_df = pd.read_pickle(os.path.join(folder, file))
        print(f"\nLoaded: {file}")
        audio_df.info()

# =====================================================
# SELECT NUMERIC FEATURES ONLY
# =====================================================

feature_cols = audio_df.select_dtypes(include=[np.number]).columns
drop_cols = ["time stamp", "duration"] # remove time columns if don't want them in embedding
feature_cols = [c for c in feature_cols if c not in drop_cols]
print("\nUsing features:")
print(feature_cols)
X_audio = audio_df[feature_cols].dropna()
# audio_clean = audio_df.loc[X_audio.index].copy() # keep alignment index (important for interpretation)

# =====================================================
# STANDARDIZE
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_audio)
print("\nAudio matrix shape:", X_scaled.shape)

# =====================================================
# PCA
# =====================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(7,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=25, alpha=0.7)
plt.title("PCA of Audio Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
save_path = os.path.join(output_folder, f"audio_pca_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()

# =====================================================
# PCA VARIANCE ANALYSIS (SCREE PLOT)
# =====================================================
pca_full = PCA().fit(X_scaled)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_var) + 1), explained_var, marker="o",  label="Explained variance")
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var,marker="s",label="Cumulative variance")
plt.title("PCA Explained Variance (Audio Features)")
plt.xlabel("Number of Components")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"audio_pca_variance_{target}.png"), dpi=300)
plt.show()

# =====================================================
# t-SNE
# =====================================================
# pca = PCA(n_components=10) # variance >0.9
# X_pca = pca.fit_transform(X_scaled)
tsne = TSNE(n_components=2, perplexity=10, random_state=42, init="pca") ## 5 7 8 10 15 30 50 (perplexity ~ dataset size)
X_tsne = tsne.fit_transform(X_scaled) ## X_scaled X_pca
plt.figure(figsize=(7,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=25, alpha=0.7)
plt.title("t-SNE of Audio Features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
save_path = os.path.join(output_folder, f"audio_tsne_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()