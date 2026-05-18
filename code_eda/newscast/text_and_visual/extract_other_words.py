import os
import pandas as pd
from collections import Counter

# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------
input_dir = "/home/matilde/Desktop/PBD/results/newscast/ocr_topics_final/"
clean_dir = "/home/matilde/Desktop/PBD/matilde_features/newscast_clean/"

# ---------------------------------------------------------
# Process each newscast folder
# ---------------------------------------------------------
for newscast in os.listdir(input_dir):

    newscast_path = os.path.join(input_dir, newscast)
    if not os.path.isdir(newscast_path):
        continue

    print(f"\n==============================")
    print(f"Analyzing OTHER segments for: {newscast}")
    print(f"==============================\n")

    # Load segmentation file
    seg_file = None
    for f in os.listdir(newscast_path):
        if f.endswith("_topic_segmentation_final.txt"):
            seg_file = os.path.join(newscast_path, f)
            break

    if seg_file is None:
        print("No segmentation file found. Skipping.")
        continue

    # Load cleaned OCR file
    clean_file = os.path.join(
        clean_dir,
        newscast.replace("_topic_segmentation_final", "") + "_clean.pkl"
    )

    if not os.path.exists(clean_file):
        print("Clean OCR file not found. Skipping.")
        continue

    df = pd.read_pickle(clean_file)

    # ---------------------------------------------------------
    # Parse segmentation file
    # ---------------------------------------------------------
    segments = []
    with open(seg_file, "r") as f:
        for line in f:
            if line.startswith("- Topic:"):
                parts = line.split(",")
                topic = parts[0].replace("- Topic:", "").strip()
                frame_info = parts[1].strip()

                start = int(frame_info.split(" ")[1])
                end = int(frame_info.split(" ")[3])

                segments.append((topic, start, end))

    # ---------------------------------------------------------
    # Extract words from OTHER segments
    # ---------------------------------------------------------
    all_other_words = []

    for topic, start, end in segments:
        if topic != "Other":
            continue

        print(f"OTHER segment: frames {start} to {end}")

        segment_words = []
        for text in df["clean_text"].iloc[start:end+1]:
            segment_words.extend(text.split())

        all_other_words.extend(segment_words)

        # Print words for this segment
        freq = Counter(segment_words).most_common(40)
        print("Top words in this OTHER segment:")
        for w, c in freq:
            print(f"  {w}: {c}")
        print("\n")

    # ---------------------------------------------------------
    # Print aggregated OTHER words
    # ---------------------------------------------------------
    print("====================================")
    print("GLOBAL TOP WORDS IN ALL OTHER SEGMENTS")
    print("====================================")

    global_freq = Counter(all_other_words).most_common(80)
    for w, c in global_freq:
        print(f"{w}: {c}")

    print("\nDone.\n")
