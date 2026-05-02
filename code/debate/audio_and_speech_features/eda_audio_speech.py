import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler # normalize with z-score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import itertools
from sklearn.metrics import r2_score

folder = 'Project_Features'

# AUDIO (single file example) - check file structure
video = 'Ventura_vs_Marques_Mendes_November_25'
data_audio = pd.read_pickle(os.path.join(folder, video + '_audio.pkl'))
data_audio.info()


# SIMILARITY BETWEEN DEBATES USING SOME RAW FEATURE EMBEDDINGS # maybe try all numeric features instead of subgroup
features = []
for file in os.listdir(folder):
    if file.endswith('_audio.pkl'):
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        features.append({
            "debate": file.replace('_audio.pkl', ''),           
            "meanF0Hz_mean": df["meanF0Hz"].mean(),  # Pitch # maybe try q1 and q3 because of each candidate
            "stdevF0Hz_mean": df["stdevF0Hz"].mean(),
            "HNR_mean": df["HNR"].mean(),  # Voice quality
            "jitter_mean": df["localJitter"].mean(),
            "speechrate_mean": df["speechrate"].mean(),  # Fluency
            "articulationrate_mean": df["articulationrate"].mean(),
            "npause_rate": df["npause"].sum() / df["duration"].sum(),
            "fdisp_mean": df["fdisp"].mean(), # Articulation
        })
audio_df = pd.DataFrame(features)
X = audio_df.drop(columns=["debate"])
X_scaled = StandardScaler().fit_transform(X)
sim = cosine_similarity(X_scaled)
plt.figure(figsize=(10, 8))
sns.heatmap(
    sim,
    cmap="Blues",
    annot=False,
    xticklabels=False,
    yticklabels=audio_df["debate"]
)
plt.title("Audio Feature Similarity (Cosine)")
plt.tight_layout()
plt.show()


# SIMILARITY BETWEEN DEBATES USING LLM EMBEDDINGS
embeddings = []
doc_names = []
for file in os.listdir(folder):
    if file.endswith("_audio.pkl"):
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        emb_matrix = np.vstack(df["speak_embeddings"].values)
        mean_emb = emb_matrix.mean(axis=0)
        embeddings.append(mean_emb)
        doc_names.append(file.replace("_audio.pkl", ""))
embeddings = np.array(embeddings)
sim = cosine_similarity(embeddings)
plt.figure(figsize=(10, 8))
sns.heatmap(
    sim,
    cmap="Blues",
    annot=False,
    xticklabels=False,
    yticklabels=doc_names
)
plt.title("Debate Similarity (Speech Embeddings)")
plt.tight_layout()
plt.show()


# CORRELATION BETWEEN FEATURES # maybe try all numeric features instead of subgroup
features = []
for file in os.listdir(folder):
    if file.endswith('_audio.pkl'):
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        features.append({
            "debate": file.replace('_audio.pkl', ''),           
            "meanF0Hz_mean": df["meanF0Hz"].mean(),  # Pitch # maybe try q1 and q3 because of each candidate
            "stdevF0Hz_mean": df["stdevF0Hz"].mean(),
            "HNR_mean": df["HNR"].mean(),  # Voice quality
            "jitter_mean": df["localJitter"].mean(),
            "speechrate_mean": df["speechrate"].mean(),  # Fluency
            "articulationrate_mean": df["articulationrate"].mean(),
            "npause_rate": df["npause"].sum() / df["duration"].sum(),
            "fdisp_mean": df["fdisp"].mean(), # Articulation
        })
audio_df = pd.DataFrame(features)
corr = audio_df.drop(columns=["debate"]).corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Audio Feature Correlations")
plt.show()


# PAIRPLOTS (aggregate per debate)
features = []
for file in os.listdir(folder):
    if file.endswith('_audio.pkl'):
        path = os.path.join(folder, file)
        df = pd.read_pickle(path)
        features.append({
            "debate": file.replace('_audio.pkl', ''),
            "meanF0Hz": df["meanF0Hz"].mean(),
            "stdevF0Hz": df["stdevF0Hz"].mean(),
            "speechrate": df["speechrate"].mean(),
            "articulationrate": df["articulationrate"].mean(),
            "HNR": df["HNR"].mean(),
            "jitter": df["localJitter"].mean(),
        })
audio_df = pd.DataFrame(features)
X = audio_df.drop(columns=["debate"])
X_scaled = StandardScaler().fit_transform(X) 
sns.pairplot(pd.DataFrame(X_scaled, columns=X.columns), height=1.5,aspect=1)
plt.show()


# REGRESSION PLOTS
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
pairs = list(itertools.combinations(X.columns, 2))
for x, y in pairs:
    x_vals = df_scaled[x].values
    y_vals = df_scaled[y].values
    lin_coef = np.polyfit(x_vals, y_vals, 1)
    y_pred_lin = lin_coef[0]*x_vals + lin_coef[1]
    r2_lin = r2_score(y_vals, y_pred_lin)
    quad_coef = np.polyfit(x_vals, y_vals, 2)
    y_pred_quad = quad_coef[0]*x_vals**2 + quad_coef[1]*x_vals + quad_coef[2]
    r2_quad = r2_score(y_vals, y_pred_quad)
    if r2_lin >= 0.5:
        plt.figure(figsize=(4,3))
        sns.regplot(x=x_vals,y=y_vals, scatter_kws={"s": 30},line_kws={"color": "red"} )
        plt.title(f"{x} vs {y} (Linear R²={r2_lin:.2f})")
        plt.tight_layout()
    if r2_quad >= 0.5 and r2_quad > r2_lin + 0.05:
        plt.figure(figsize=(4,3))
        sns.scatterplot(x=x_vals, y=y_vals)
        xs = np.linspace(x_vals.min(), x_vals.max(), 100)
        ys = quad_coef[0]*xs**2 + quad_coef[1]*xs + quad_coef[2]
        plt.plot(xs, ys, color="red")
        plt.title(f"{x} vs {y} (Quadratic R²={r2_quad:.2f})")
        plt.tight_layout()
plt.show()


# TIMESERIES (single debate) - speechrate
df = data_audio.sort_values("time stamp")
plt.figure(figsize=(10,4))
plt.plot(df["time stamp"], df["speechrate"], alpha=0.3)
df["speechrate_smooth"] = df["speechrate"].rolling(window=5, center=True).mean()
plt.plot(df["time stamp"], df["speechrate_smooth"])
plt.title("Speech Rate Over Time - Ventura_vs_Marques_Mendes_November_25")
plt.xlabel("Time (s)")
plt.ylabel("Speech Rate")
plt.show()


# TIMESERIES (single debate) - meanF0Hz
plt.figure(figsize=(10,4))
plt.plot(df["time stamp"], df["meanF0Hz"], alpha=0.3)
df["f0_smooth"] = df["meanF0Hz"].rolling(window=5, center=True).mean()
plt.plot(df["time stamp"], df["f0_smooth"])
plt.title("Pitch (F0) Over Time - Ventura_vs_Marques_Mendes_November_25")
plt.xlabel("Time (s)")
plt.ylabel("F0 Hz")
plt.show()