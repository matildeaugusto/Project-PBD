# TODO: Concatenate face and pose (visual) with audio by concatenating PCA_30's of each modality
# TODO: Clustering

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import ast
import plotly.express as px

def save_plotly_audio_scatter(points,time_stamps,durations,title,save_html,labels=None,search_time=None):
    # TODO: Add search_times to see who speaks
    df_plot = pd.DataFrame({"x": points[:, 0],"y": points[:, 1],"time": time_stamps, "duration": durations})
    fig = px.scatter(df_plot, x="x", y="y",color=labels if labels is not None else None, hover_data=["time", "duration"], title=title,opacity=0.4)
    fig.update_traces(marker=dict(size=4))
    if search_time is not None:
        if isinstance(search_time, (int, float)):
            search_time = [search_time]
        mask = np.zeros(len(df_plot), dtype=bool)
        for t in search_time:
            mask |= np.isclose(df_plot["time"], t, atol=0.05)
        df_match = df_plot[mask]
        fig.add_scatter(x=df_match["x"],y=df_match["y"],mode="markers",name=f"Matches: {search_time}",marker=dict(size=10, color="red"),hovertext=df_match["time"].astype(str))
    fig.write_html(save_html)
    print(f"Saved: {save_html}")

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
# SELECT NUMERIC FEATURES ONLY - Raw Audio Features
# =====================================================
# audio_df["npause"] = audio_df["npause"].astype(float) ## 
# audio_df["pause_rate"] = audio_df["npause"] / (audio_df["duration"] + 1e-6) ## 
feature_cols = audio_df.select_dtypes(include=[np.number]).columns
drop_cols = ["time stamp", "duration"] ## , "npause"
feature_cols = [c for c in feature_cols if c not in drop_cols]
print("\nUsing features:")
print(feature_cols)
X_audio = audio_df[feature_cols].dropna()
audio_df["time"] = audio_df["time stamp"].astype(float) ## 
audio_df["duration"] = audio_df["duration"].astype(float) ##
audio_df["end_time"] = audio_df["time"] + audio_df["duration"] ##

# =====================================================
# STANDARDIZE - Raw Audio Features
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_audio)
print("\nAudio matrix shape:", X_scaled.shape)

# =====================================================
# PCA - Raw Audio Features
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
save_plotly_audio_scatter( X_pca,audio_df["time"].values,audio_df["duration"].values,"PCA of Audio Features",os.path.join(output_folder, f"audio_pca_{target}.html"))
plt.show()

# =====================================================
# PCA VARIANCE ANALYSIS - Raw Audio Features
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
# t-SNE - Raw Audio Features
# =====================================================
pca = PCA(n_components=20) ## variance >0.9
X_pca = pca.fit_transform(X_scaled) ## 
tsne = TSNE(n_components=2, perplexity=7, random_state=42, init="pca") ## 5 "7" 8 9 10 15 30 50 (perplexity ~ dataset size)
X_tsne = tsne.fit_transform(X_pca) ## X_scaled X_pca
plt.figure(figsize=(7,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=25, alpha=0.7)
plt.title("t-SNE of Audio Features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
save_path = os.path.join(output_folder, f"audio_tsne_{target}.png")
plt.savefig(save_path, dpi=300)
save_plotly_audio_scatter(X_tsne,audio_df["time"].values,audio_df["duration"].values,"PCA of Audio Features",os.path.join(output_folder, f"audio_tsne_{target}.html"))
plt.show()


# =====================================================
# Embeddings Plots
# =====================================================
embeddings = []
for e in audio_df["speak_embeddings"]:
    if e is None:
        continue
    if isinstance(e, float) and np.isnan(e):
        continue
    if isinstance(e, str):
        try:
            e = ast.literal_eval(e)
        except:
            continue
    if isinstance(e, (list, np.ndarray)):
        embeddings.append(np.array(e).flatten())
X_emb = np.vstack(embeddings)
scaler_emb = StandardScaler()
X_emb_scaled = scaler_emb.fit_transform(X_emb)

pca_emb = PCA(n_components=2)
X_emb_pca = pca_emb.fit_transform(X_emb_scaled)
plt.figure(figsize=(7,6))
plt.scatter(X_emb_pca[:, 0], X_emb_pca[:, 1], s=25, alpha=0.7)
plt.title("PCA of Speech Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(output_folder, f"audio_embeddings_pca_{target}.png"), dpi=300)
save_plotly_audio_scatter(X_emb_pca,audio_df["time"].values,audio_df["duration"].values,"PCA of Audio Embeddings",os.path.join(output_folder, f"audio_embedings_pca_{target}.html"))
plt.show()

pca_emb_full = PCA().fit(X_emb_scaled)
explained_var_emb = pca_emb_full.explained_variance_ratio_
cumulative_var_emb = np.cumsum(explained_var_emb)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_var_emb) + 1),explained_var_emb, marker="o", label="Explained variance")
plt.plot(range(1, len(cumulative_var_emb) + 1),cumulative_var_emb,marker="s",label="Cumulative variance")
plt.title("PCA Explained Variance (Speech Embeddings)")
plt.xlabel("Number of Components")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"audio_embeddings_pca_variance_{target}.png"),dpi=300)
plt.show()

# X_emb_scaled_pre = X_emb_scaled ## 
pca_pre = PCA(n_components=0.85) ## 0.85 variance
X_emb_scaled_pre = pca_pre.fit_transform(X_emb_scaled) ## 
print("PCA components used:", pca_pre.n_components_) ## 
print("Explained variance:", np.sum(pca_pre.explained_variance_ratio_)) ## 
tsne_emb = TSNE(n_components=2,perplexity=10,random_state=42, init="pca")
X_emb_tsne = tsne_emb.fit_transform(X_emb_scaled_pre) ##  X_emb_scaled_pre X_emb_scaled
print("Emb Matrix t-SNE shape:", X_emb_tsne.shape)
plt.figure(figsize=(7,6))
plt.scatter(X_emb_tsne[:, 0], X_emb_tsne[:, 1], s=25, alpha=0.7)
plt.title("t-SNE of Speech Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.savefig(os.path.join(output_folder, f"audio_embeddings_tsne_{target}.png"), dpi=300)
save_plotly_audio_scatter(X_emb_tsne,audio_df["time"].values,audio_df["duration"].values,"t-SNE of Audio Embeddings",os.path.join(output_folder, f"audio_embedings_tsne_{target}.html"))
plt.show()