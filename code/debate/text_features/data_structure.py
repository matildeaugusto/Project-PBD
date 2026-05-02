import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Directory where results will be saved
# ---------------------------------------------------------
output_dir = "/home/matilde/Desktop/PBD/results/debates/data_structure"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Helper function to detect columns containing vectors/arrays
# ---------------------------------------------------------
def detect_vector_columns(df):
    vector_columns = []
    for col in df.columns:
        try:
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                vector_columns.append(col)
        except:
            pass
    return vector_columns


# ---------------------------------------------------------
# Main function to analyze a .pkl file
# ---------------------------------------------------------
def analyze_file(path):

    # Create TXT filename inside the results directory
    base_name = os.path.basename(path).replace(".pkl", "")
    output_file = os.path.join(output_dir, f"{base_name}_analysis.txt")

    with open(output_file, 'w') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Analyzing file: {path}\n")
        f.write("="*80 + "\n")

        try:
            df = pd.read_pickle(path)
        except Exception as e:
            f.write(f"Error loading {path}: {e}\n")
            return

        # Shape
        f.write(f"\nShape: {df.shape}\n")

        # Export to Excel (same directory)
        excel_file = os.path.join(output_dir, f"{base_name}_data.xlsx")
        df.to_excel(excel_file, index=False)
        f.write(f"\nData exported to Excel: {excel_file}\n")

        # Columns
        f.write("\nColumns:\n")
        f.write(str(df.columns.tolist()) + "\n")

        # First rows
        f.write("\nFirst rows:\n")
        f.write(str(df.head()) + "\n")

        # Data types
        f.write("\nData types:\n")
        f.write(str(df.dtypes) + "\n")

        # Missing values
        f.write("\nMissing values per column:\n")
        f.write(str(df.isnull().sum()) + "\n")

        # Timestamp detection
        for col in df.columns:
            if "time" in col.lower() or "timestamp" in col.lower():
                try:
                    f.write(f"\nTimestamp column detected: {col}\n")
                    f.write(f"   Range: {df[col].min()} to {df[col].max()}\n")
                except:
                    f.write(f"\nTimestamp column detected but range could not be computed: {col}\n")

        # Relevant IDs
        ids = ["frame_id", "segment_id", "speaker_id", "person_id"]
        for id_col in ids:
            if id_col in df.columns:
                f.write(f"ID column detected: {id_col}\n")

        # Vector columns
        vector_cols = detect_vector_columns(df)
        if vector_cols:
            f.write("\nColumns containing vectors/arrays:\n")
            for col in vector_cols:
                f.write(f"   - {col}\n")

        f.write("\nAnalysis completed for this file.\n")

    print(f"TXT saved to: {output_file}")
    print(f"Excel saved to: {excel_file}")


# ---------------------------------------------------------
# Files to analyze
# ---------------------------------------------------------
base_path = "/home/matilde/Desktop/PBD/matilde_features/debates/"

files = [
    base_path + "candidate_party.pkl",
    base_path + "polls_timeline.pkl",
]

# ---------------------------------------------------------
# Run analysis
# ---------------------------------------------------------
for f in files:
    if os.path.exists(f):
        analyze_file(f)
    else:
        print(f"File not found: {f}")
