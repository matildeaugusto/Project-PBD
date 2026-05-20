# TODO: Change to not be the mode (use all frame so propotion of duration matters)
# TODO: Change to not be just frames with 1 person

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from scipy.stats import kruskal
import scipy.stats as ss


# ----------------------------------------------------------------------------------------------------------------------
# CONFIG

folder = 'Project_Features'
target = "Marques_Mendes_vs_Gouveia_Melo_December_21"
output_folder = "part2_v05"

features = [
    'meanF0Hz',
    'stdevF0Hz',
    'speechrate',
    'articulationrate',
    'HNR',
    'localJitter'
]

# ----------------------------------------------------------------------------------------------------------------------
# LOAD FILES (ONE DEBATE)

visual_df, audio_df = None, None

for file in os.listdir(folder):
    if file.endswith('_visual.pkl'):
        if target in file:
            visual_df = pd.read_pickle(os.path.join(folder, file))
            print("\nVISUAL FILE:", file)
            visual_df.info()

for file in os.listdir(folder):
    if file.endswith('_audio.pkl'):
        if target in file:
            audio_df = pd.read_pickle(os.path.join(folder, file))
            print("\nAUDIO FILE:", file)
            audio_df.info()

# ----------------------------------------------------------------------------------------------------------------------
# CLEAN VISUAL TIME

def extract_frame_number(path):
    match = re.search(r'frame_(\d+)', path)
    return int(match.group(1)) if match else None

visual_df['frame_id'] = visual_df['Frame'].apply(extract_frame_number)
visual_df = visual_df.dropna(subset=['frame_id'])
visual_df['frame_id'] = visual_df['frame_id'].astype(int)
visual_df['time'] = visual_df['frame_id']

# ----------------------------------------------------------------------------------------------------------------------
# ALIGN VISUAL - AUDIO (FIXED)
single_face_frames = 0
multi_face_frames = 0
empty_frames = 0
audio_df['end_time'] = audio_df['time stamp'] + audio_df['duration']
segment_emotions = []
for _, row in audio_df.iterrows():
    start, end = row['time stamp'], row['end_time']
    frames = visual_df[
        (visual_df['time'] >= start) &
        (visual_df['time'] <= end)
    ]
    emotions = []
    for fer in frames['Fer']:
        if isinstance(fer, dict):
            fer = [fer]
        if not isinstance(fer, list) or len(fer) == 0:
            continue

        if len(fer) == 1:
            emotions.append(fer[0]["top_emotion"])
        elif len(fer) > 1:
            continue

    if len(emotions) == 0:
        segment_emotions.append(None)
    else:
        segment_emotions.append(pd.Series(emotions).mode()[0])



audio_df['dominant_emotion'] = segment_emotions


# ----------------------------------------------------------------------------------------------------------------------
# CLEAN DATA

df = audio_df[features + ['dominant_emotion']].dropna()
print(df["dominant_emotion"].value_counts())
df = df[df["dominant_emotion"].isin(
 df["dominant_emotion"].value_counts()[df["dominant_emotion"].value_counts() > 5].index ## >5
)] 

# ----------------------------------------------------------------------------------------------------------------------
# DISTRIBUTIONS (BOXPLOTS)

for f in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='dominant_emotion', y=f)
    plt.xticks(rotation=45)
    plt.title(f"{f} by Emotion")
    plt.tight_layout()
    save_path = os.path.join(output_folder, f"emotions_audio_boxplot_{f}_{target}.png")
    plt.savefig(save_path, dpi=300)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# KRUSKAL-WALLIS TEST

print("\nKruskal-Wallis tests:")

for f in features:

    groups = [
        df[df['dominant_emotion'] == e][f]
        for e in df['dominant_emotion'].dropna().unique()
    ]

    stat = kruskal(*groups)
    print(f"{f}: p-value = {stat.pvalue:.4f}")


# ----------------------------------------------------------------------------------------------------------------------
# CHI-SQUARE CONTINGENCY TEST

max_bins = 6
bin_results = []
for f in features:
    temp = df[[f, "dominant_emotion"]].dropna()
    feature_row = {"feature": f}
    for bins in range(2, max_bins+1): 
        try:
            if temp[f].nunique() < bins:
                feature_row[f"{bins}_bins"] = np.nan
                continue
            temp[f + "_level"] = pd.qcut(
                temp[f],
                q=bins,
                duplicates="drop",
                labels=False
            )
            cont_tab = pd.crosstab(
                temp[f + "_level"],
                temp["dominant_emotion"]
            )
            if cont_tab.shape[0] < 2 or cont_tab.shape[1] < 2:
                feature_row[f"{bins}_bins"] = np.nan
                continue
            stat, p, dof, expected = ss.chi2_contingency(cont_tab)
            critical = ss.chi2.ppf(0.95, dof)
            feature_row[f"{bins}_bins"] = int(stat > critical)
            n = cont_tab.to_numpy().sum()
            k = min(cont_tab.shape)
            cramers_v = np.sqrt(stat / (n * (k - 1)))
        except:
            feature_row[f"{bins}_bins"] = np.nan
    bin_results.append(feature_row)
results_df = pd.DataFrame(bin_results).set_index("feature")
plt.figure(figsize=(8,5))
sns.heatmap(
    results_df,
    annot=True,
    cmap="RdYlGn_r",
    cbar=False,
    linewidths=0.5
)
plt.title("Chi-Square H0 Rejection Stability (Feature vs. Bins)")
plt.xlabel("Number of bins")
plt.ylabel("Audio features")
plt.tight_layout()
save_path = os.path.join(output_folder, f"emotions_audio_chi-square_h0-rejection_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()

bin_results = []
for f in features:
    temp = df[[f, "dominant_emotion"]].dropna()
    feature_row = {"feature": f}
    for bins in range(2, max_bins+1):
        try:
            if temp[f].nunique() < bins:
                feature_row[f"{bins}_bins"] = np.nan
                continue
            temp[f + "_level"] = pd.qcut(
                temp[f],
                q=bins,
                duplicates="drop",
                labels=False
            )
            cont_tab = pd.crosstab(
                temp[f + "_level"],
                temp["dominant_emotion"]
            )
            if cont_tab.shape[0] < 2 or cont_tab.shape[1] < 2:
                feature_row[f"{bins}_bins"] = np.nan
                continue
            stat, p, dof, expected = ss.chi2_contingency(cont_tab)
            feature_row[f"{bins}_bins"] = p
        except:
            feature_row[f"{bins}_bins"] = np.nan
    bin_results.append(feature_row)
results_df = pd.DataFrame(bin_results).set_index("feature")
plt.figure(figsize=(9,5))
sns.heatmap(
   results_df,  
    annot=results_df.round(3),
    cmap="viridis",
    linewidths=0.5
)
plt.title("Chi-Square p-values (Feature vs. Bins)")
plt.xlabel("Number of bins")
plt.ylabel("Audio features")
plt.tight_layout()
save_path = os.path.join(output_folder, f"emotions_audio_chi-square_p-values_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()
