import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram

# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------
input_root = "/home/matilde/Desktop/PBD/results/newscast/ocr_topics_final_clean/"
output_root = "/home/matilde/Desktop/PBD/results/newscast/global_analysis/"
os.makedirs(output_root, exist_ok=True)

# Subfolders for images
img_day_dir = os.path.join(output_root, "by_day")
img_channel_dir = os.path.join(output_root, "by_channel")
img_day_channel_dir = os.path.join(output_root, "by_day_channel")
img_heatmaps = os.path.join(output_root, "heatmaps")
img_similarity = os.path.join(output_root, "similarity")

os.makedirs(img_day_dir, exist_ok=True)
os.makedirs(img_channel_dir, exist_ok=True)
os.makedirs(img_day_channel_dir, exist_ok=True)
os.makedirs(img_heatmaps, exist_ok=True)
os.makedirs(img_similarity, exist_ok=True)

# ---------------------------------------------------------
# Load all final JSON files
# ---------------------------------------------------------
json_files = [f for f in os.listdir(input_root) if f.endswith("_final.json")]

if len(json_files) == 0:
    print("No final JSON files found. Run the segmentation script first.")
    exit()

records = []

for jf in json_files:
    with open(os.path.join(input_root, jf), "r") as f:
        data = json.load(f)

    newscast = data["newscast"]
    channel = data["channel"]
    date = data["date"]

    for topic, info in data["topics"].items():
        records.append({
            "newscast": newscast,
            "channel": channel,
            "date": date,
            "topic": topic,
            "percent": info["percent"]
        })

df = pd.DataFrame(records)

# ---------------------------------------------------------
# Helper: bar chart
# ---------------------------------------------------------
def plot_bar(series, title, filename):
    plt.figure(figsize=(10,6))
    series.plot(kind="bar", color="steelblue")
    plt.title(title)
    plt.ylabel("Percent (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ---------------------------------------------------------
# 1) Top topics per day
# ---------------------------------------------------------
days = df["date"].unique()

for day in days:
    day_df = df[df["date"] == day]
    grouped = day_df.groupby("topic")["percent"].mean().sort_values(ascending=False)

    filename = os.path.join(img_day_dir, f"top_topics_day_{day}.png")
    plot_bar(grouped, f"Top Topics on {day}", filename)

# ---------------------------------------------------------
# 2) Top topics per channel
# ---------------------------------------------------------
channels = df["channel"].unique()

for ch in channels:
    ch_df = df[df["channel"] == ch]
    grouped = ch_df.groupby("topic")["percent"].mean().sort_values(ascending=False)

    filename = os.path.join(img_channel_dir, f"top_topics_channel_{ch}.png")
    plot_bar(grouped, f"Top Topics on Channel {ch}", filename)

# ---------------------------------------------------------
# 3) Top topics per channel per day
# ---------------------------------------------------------
for ch in channels:
    for day in days:
        subset = df[(df["channel"] == ch) & (df["date"] == day)]
        if len(subset) == 0:
            continue

        grouped = subset.groupby("topic")["percent"].mean().sort_values(ascending=False)

        filename = os.path.join(img_day_channel_dir, f"top_topics_{ch}_{day}.png")
        plot_bar(grouped, f"Top Topics for {ch} on {day}", filename)

# ---------------------------------------------------------
# 4) Heatmap: channel × topic
# ---------------------------------------------------------
pivot_channel = df.pivot_table(index="channel", columns="topic", values="percent", aggfunc="mean").fillna(0)

plt.figure(figsize=(12,6))
sns.heatmap(pivot_channel, annot=True, cmap="Blues")
plt.title("Heatmap: Topic × Channel")
plt.tight_layout()
plt.savefig(os.path.join(img_heatmaps, "heatmap_channel_topic.png"))
plt.close()

# ---------------------------------------------------------
# 5) Heatmap: day × topic
# ---------------------------------------------------------
pivot_day = df.pivot_table(index="date", columns="topic", values="percent", aggfunc="mean").fillna(0)

plt.figure(figsize=(12,6))
sns.heatmap(pivot_day, annot=True, cmap="Greens")
plt.title("Heatmap: Topic × Day")
plt.tight_layout()
plt.savefig(os.path.join(img_heatmaps, "heatmap_day_topic.png"))
plt.close()

# ---------------------------------------------------------
# 6) Channel similarity (cosine similarity)
# ---------------------------------------------------------
sim_matrix = cosine_similarity(pivot_channel)
sim_df = pd.DataFrame(sim_matrix, index=pivot_channel.index, columns=pivot_channel.index)

plt.figure(figsize=(8,6))
sns.heatmap(sim_df, annot=True, cmap="Purples")
plt.title("Channel Similarity (Cosine Similarity)")
plt.tight_layout()
plt.savefig(os.path.join(img_similarity, "channel_similarity.png"))
plt.close()

# ---------------------------------------------------------
# 7) Hierarchical clustering of channels
# ---------------------------------------------------------
linkage_matrix = linkage(pivot_channel, method="ward")

plt.figure(figsize=(8,6))
dendrogram(linkage_matrix, labels=pivot_channel.index)
plt.title("Hierarchical Clustering of Channels")
plt.tight_layout()
plt.savefig(os.path.join(img_similarity, "channel_clustering.png"))
plt.close()

# ---------------------------------------------------------
# 8) Topic diversity (entropy)
# ---------------------------------------------------------
def entropy(values):
    p = np.array(values) / np.sum(values)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

diversity = {}

for ch in channels:
    ch_df = df[df["channel"] == ch]
    topic_means = ch_df.groupby("topic")["percent"].mean()
    diversity[ch] = entropy(topic_means.values)

diversity_series = pd.Series(diversity).sort_values(ascending=False)

plot_bar(diversity_series, "Topic Diversity per Channel (Entropy)", os.path.join(output_root, "diversity_channels.png"))

# ---------------------------------------------------------
# 9) Global agenda share
# ---------------------------------------------------------
agenda_share = df.groupby("topic")["percent"].mean().sort_values(ascending=False)

plot_bar(agenda_share, "Global Agenda Share (Average Percent)", os.path.join(output_root, "agenda_share_global.png"))

# ---------------------------------------------------------
# 10) Global text report
# ---------------------------------------------------------
report_path = os.path.join(output_root, "global_topic_report.txt")

with open(report_path, "w") as f:

    f.write("=====================================\n")
    f.write("GLOBAL TOPIC ANALYSIS (NORMALIZED)\n")
    f.write("=====================================\n\n")

    f.write("GLOBAL AGENDA SHARE:\n")
    for topic, pct in agenda_share.items():
        f.write(f"  {topic}: {pct:.1f}%\n")

    f.write("\nTOPIC DIVERSITY (ENTROPY):\n")
    for ch, val in diversity.items():
        f.write(f"  {ch}: {val:.3f}\n")

    f.write("\nCHANNEL SIMILARITY MATRIX:\n")
    f.write(sim_df.to_string())
    f.write("\n")

print("Global analysis complete.")
print(f"Results saved to: {output_root}")
