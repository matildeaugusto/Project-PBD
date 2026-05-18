import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Directory where results will be saved
# ---------------------------------------------------------
output_dir = "/home/matilde/Desktop/PBD/results/debates/supporting_party_EDA/"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
input_file = "/home/matilde/Desktop/PBD/matilde_features/debates/candidate_party.pkl"
df = pd.read_pickle(input_file)

# ---------------------------------------------------------
# Create TXT report (EDA only, no data structure repetition)
# ---------------------------------------------------------
report_path = os.path.join(output_dir, "supporting_party_EDA.txt")

with open(report_path, "w") as f:

    f.write("\n==============================\n")
    f.write("Supporting Party - EDA Report\n")
    f.write("==============================\n\n")

    # Number of unique candidates
    f.write("Number of candidates: ")
    f.write(str(df['Candidate'].nunique()) + "\n\n")

    # Number of unique parties
    f.write("Number of parties: ")
    f.write(str(df['Party'].nunique()) + "\n\n")

    # Count candidates per party
    f.write("Candidates per party:\n")
    f.write(str(df['Party'].value_counts()) + "\n\n")

    # Party proportions
    f.write("Party proportions (%):\n")
    proportions = df['Party'].value_counts(normalize=True) * 100
    f.write(str(proportions.round(2)) + "\n\n")

    # Candidate → Party mapping
    party_map = dict(zip(df['Candidate'], df['Party']))
    f.write("Candidate to Party mapping:\n")
    f.write(str(party_map) + "\n\n")

    # Uniqueness check: one candidate per party?
    f.write("Number of candidates per party (uniqueness check):\n")
    f.write(str(df.groupby("Party")["Candidate"].nunique()) + "\n\n")

    # Normalized candidate names (useful for merging with other modalities)
    f.write("Normalized candidate names (lowercase, no underscores):\n")
    normalized = df['Candidate'].apply(lambda x: x.lower().replace("_", " "))
    f.write(str(normalized.tolist()) + "\n\n")

    # Coalition detection
    f.write("Coalition vs Single Party classification:\n")
    coalition_check = df['Party'].apply(lambda x: "Coalition" if "+" in x else "Single Party")
    f.write(str(coalition_check.value_counts()) + "\n\n")

    # Party index (useful for merges and ordering)
    df['Party_Index'] = df['Party'].astype('category').cat.codes
    f.write("Party index mapping:\n")
    f.write(str(df[['Party', 'Party_Index']].drop_duplicates()) + "\n\n")

    # Summary table
    f.write("Summary table (Party, Num_Candidates):\n")
    summary = df['Party'].value_counts().reset_index()
    summary.columns = ['Party', 'Num_Candidates']
    f.write(str(summary) + "\n\n")

print(f"EDA report saved to: {report_path}")

# ---------------------------------------------------------
# Visualizations
# ---------------------------------------------------------

# Bar chart: number of candidates per party
plt.figure(figsize=(8, 5))
df['Party'].value_counts().plot(kind='bar', color='steelblue')
plt.title("Number of Candidates per Party")
plt.xlabel("Party")
plt.ylabel("Count")
plt.tight_layout()
bar_plot_path = os.path.join(output_dir, "party_bar_chart.png")
plt.savefig(bar_plot_path)
plt.close()

# Pie chart: party distribution
plt.figure(figsize=(6, 6))
df['Party'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Party Distribution")
plt.ylabel("")
plt.tight_layout()
pie_plot_path = os.path.join(output_dir, "party_pie_chart.png")
plt.savefig(pie_plot_path)
plt.close()

print(f"Bar chart saved to: {bar_plot_path}")
print(f"Pie chart saved to: {pie_plot_path}")

print("Supporting Party EDA completed successfully.")
