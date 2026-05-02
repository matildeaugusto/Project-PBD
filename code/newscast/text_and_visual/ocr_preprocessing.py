import os
import pandas as pd
import ast
import re
import nltk
from nltk.corpus import stopwords

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
input_dir = "/home/matilde/Desktop/PBD/matilde_features/newscast/"
output_dir = "/home/matilde/Desktop/PBD/matilde_features/newscast_clean/"
os.makedirs(output_dir, exist_ok=True)

# Download stopwords if needed
nltk.download("stopwords")

stopwords_pt = set(stopwords.words("portuguese"))
stopwords_en = set(stopwords.words("english"))
stopwords_all = stopwords_pt.union(stopwords_en)

# Additional noise words from TV graphics
noise_words = {
    "tvi", "rtp", "sic", "jornal", "nacional", "noticias",
    "2026", "2024", "presidenciais", "telejornal"
}

# ---------------------------------------------------------
# Helper: parse OCR cell
# ---------------------------------------------------------
def parse_ocr_cell(cell):
    if isinstance(cell, list):
        return cell
    try:
        return ast.literal_eval(cell)
    except:
        return []

# ---------------------------------------------------------
# Helper: clean text
# ---------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove HTML-like tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and noise
    tokens = [
        t for t in tokens
        if t not in stopwords_all
        and t not in noise_words
        and len(t) > 2
    ]

    return " ".join(tokens)

# ---------------------------------------------------------
# Process each OCR file
# ---------------------------------------------------------
for file in os.listdir(input_dir):
    if not file.endswith("_ocr.pkl"):
        continue

    file_path = os.path.join(input_dir, file)
    df = pd.read_pickle(file_path)

    df["OCR_parsed"] = df["OCR"].apply(parse_ocr_cell)

    # Extract and clean text
    cleaned_text = []
    for row in df["OCR_parsed"]:
        row_text = []
        for box in row:
            if "text" in box:
                row_text.append(clean_text(box["text"]))
        cleaned_text.append(" ".join(row_text))

    df["clean_text"] = cleaned_text

    # Save cleaned file
    output_path = os.path.join(output_dir, file.replace("_ocr.pkl", "_clean.pkl"))
    df[["Frame", "clean_text"]].to_pickle(output_path)

    print(f"Cleaned OCR saved: {output_path}")

print("OCR preprocessing completed.")
