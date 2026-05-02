import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# ---------------------------------------------------------
# Directory where results will be saved
# ---------------------------------------------------------
output_dir = "/home/matilde/Desktop/PBD/results/debates/polls_EDA/"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
input_file = "/home/matilde/Desktop/PBD/matilde_features/debates/polls_timeline.pkl"
df = pd.read_pickle(input_file)

# ---------------------------------------------------------
# Fix date format: convert "October_13" → datetime
# ---------------------------------------------------------
def parse_date(x):
    try:
        month, day = x.split("_")
        return pd.to_datetime(f"{month} {day} 2025")
    except:
        return pd.NaT

df["parsed_date"] = df["date"].apply(parse_date)
df = df.sort_values("parsed_date")

# Identify candidate columns
candidate_cols = [c for c in df.columns if c not in ["date", "parsed_date", "Poll_source"]]

# ---------------------------------------------------------
# Create TXT report (EDA only)
# ---------------------------------------------------------
report_path = os.path.join(output_dir, "polls_EDA.txt")

with open(report_path, "w") as f:

    f.write("\n==============================\n")
    f.write("Advanced Polls Timeline - EDA Report\n")
    f.write("==============================\n\n")

    # Number of polls
    f.write(f"Number of polls: {len(df)}\n\n")

    # Poll sources
    f.write("Poll sources and counts:\n")
    f.write(str(df['Poll_source'].value_counts()) + "\n\n")

    # Average poll results per candidate
    f.write("Average poll percentage per candidate:\n")
    f.write(str(df[candidate_cols].mean().round(2)) + "\n\n")

    # Last poll per candidate
    f.write("Last poll value per candidate:\n")
    last_poll = df.sort_values("parsed_date").iloc[-1][candidate_cols].astype(float)
    f.write(str(last_poll.round(2)) + "\n\n")


    # Standard deviation (volatility)
    f.write("Volatility (standard deviation) per candidate:\n")
    f.write(str(df[candidate_cols].std().round(2)) + "\n\n")

    # Range (max - min)
    f.write("Range (max - min) per candidate:\n")
    f.write(str((df[candidate_cols].max() - df[candidate_cols].min()).round(2)) + "\n\n")

    # Ranking per poll
    f.write("Ranking per poll (highest to lowest):\n")
    ranking_table = df[candidate_cols].rank(axis=1, ascending=False, method='min')
    f.write(str(ranking_table) + "\n\n")

    # Average ranking
    f.write("Average ranking across all polls:\n")
    avg_rank = ranking_table.mean().sort_values()
    f.write(str(avg_rank.round(2)) + "\n\n")

    # Trend (slope) per candidate
    f.write("Trend (slope) per candidate:\n")
    slopes = {}
    x = np.arange(len(df))
    for c in candidate_cols:
        slope, intercept, r, p, se = linregress(x, df[c])
        slopes[c] = slope
    f.write(str(slopes) + "\n\n")

    # Pearson correlation (all polls)
    f.write("Pearson correlation between candidates:\n")
    f.write(str(df[candidate_cols].corr().round(2)) + "\n\n")

    # Spearman correlation (all polls)
    f.write("Spearman correlation between candidates:\n")
    f.write(str(df[candidate_cols].corr(method='spearman').round(2)) + "\n\n")

    # Correlation by poll source
    f.write("Correlation by poll source (separate methodologies):\n")
    for source in df['Poll_source'].unique():
        f.write(f"\nSource: {source}\n")
        sub = df[df['Poll_source'] == source]
        if len(sub) > 1:
            f.write(str(sub[candidate_cols].corr().round(2)) + "\n")
        else:
            f.write("Not enough data for correlation.\n")

    # Cross-correlation (lag 1)
    f.write("\nCross-correlation (lag 1) between candidates:\n")
    cross_corr = {}
    for c1 in candidate_cols:
        for c2 in candidate_cols:
            if c1 != c2:
                corr = np.corrcoef(df[c1][1:], df[c2][:-1])[0, 1]
                cross_corr[(c1, c2)] = round(corr, 3)
    f.write(str(cross_corr) + "\n\n")

print(f"EDA report saved to: {report_path}")

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------

# 1. Line plot: poll evolution over time
plt.figure(figsize=(10, 6))
for c in candidate_cols:
    plt.plot(df['parsed_date'], df[c], marker='o', label=c)

plt.title("Poll Evolution Over Time")
plt.xlabel("Date")
plt.ylabel("Poll Percentage")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "polls_line_plot.png"))
plt.close()

# 2. Average poll percentage
plt.figure(figsize=(8, 5))
df[candidate_cols].mean().sort_values().plot(kind='barh', color='steelblue')
plt.title("Average Poll Percentage per Candidate")
plt.xlabel("Average Percentage")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "polls_average_bar.png"))
plt.close()

# 3. Volatility
plt.figure(figsize=(8, 5))
df[candidate_cols].std().sort_values().plot(kind='barh', color='darkred')
plt.title("Volatility (Standard Deviation) per Candidate")
plt.xlabel("Std Dev")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "polls_volatility.png"))
plt.close()

# 4. Heatmap Pearson
plt.figure(figsize=(8, 6))
sns.heatmap(df[candidate_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Pearson Correlation Between Candidates")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "polls_correlation_pearson.png"))
plt.close()

# 5. Heatmap Spearman
plt.figure(figsize=(8, 6))
sns.heatmap(df[candidate_cols].corr(method='spearman'), annot=True, cmap="Greens", fmt=".2f")
plt.title("Spearman Correlation Between Candidates")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "polls_correlation_spearman.png"))
plt.close()

print("Advanced Polls EDA completed successfully.")
