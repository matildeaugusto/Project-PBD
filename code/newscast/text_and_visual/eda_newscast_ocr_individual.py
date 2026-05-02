import os
import json
import pandas as pd
from collections import Counter, defaultdict

# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------
input_dir = "/home/matilde/Desktop/PBD/matilde_features/newscast_clean/"
output_root = "/home/matilde/Desktop/PBD/results/newscast/ocr_topics_final_clean/"
os.makedirs(output_root, exist_ok=True)

# ---------------------------------------------------------
# Load external dictionary
# ---------------------------------------------------------
with open("topic_dictionary.json", "r") as f:
    TOPIC_KEYWORDS = json.load(f)

IGNORE_TOPICS = {"Noise", "Names"}

# ---------------------------------------------------------
# Extract channel and date
# ---------------------------------------------------------
def extract_channel(name):
    name = name.lower()
    if "rtp" in name:
        return "RTP"
    if "sic" in name:
        return "SIC"
    if "tvi" in name:
        return "TVI"
    return "UNKNOWN"

def extract_date(name):
    parts = name.split("_")
    return parts[-2] + "_" + parts[-1] if len(parts) >= 2 else "UNKNOWN"

# ---------------------------------------------------------
# Classify topic based on dictionary
# ---------------------------------------------------------
def classify_topic(text):
    scores = {topic: 0 for topic in TOPIC_KEYWORDS if topic not in IGNORE_TOPICS}

    for topic, keywords in TOPIC_KEYWORDS.items():
        if topic in IGNORE_TOPICS:
            continue
        for kw in keywords:
            if kw in text:
                scores[topic] += 1

    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else "Other"

# ---------------------------------------------------------
# LIGHT smoothing (remove only noise)
# ---------------------------------------------------------
def smooth_segments_light(segments):
    cleaned = []

    for i, seg in enumerate(segments):
        topic = seg["topic"]
        start = seg["start"]
        end = seg["end"]
        size = end - start + 1

        if size == 1:
            continue

        if size <= 3 and topic == "Other":
            continue

        if i > 0 and i < len(segments) - 1:
            prev_topic = segments[i - 1]["topic"]
            next_topic = segments[i + 1]["topic"]
            if topic != prev_topic and prev_topic == next_topic and size <= 3:
                continue

        cleaned.append(seg)

    return cleaned

# ---------------------------------------------------------
# Count topic blocks
# ---------------------------------------------------------
def count_topic_blocks(segments):
    blocks = []
    last_topic = None

    for seg in segments:
        topic = seg["topic"]
        if topic != last_topic:
            blocks.append(topic)
        last_topic = topic

    return Counter(blocks)

# ---------------------------------------------------------
# Compute total frames per topic
# ---------------------------------------------------------
def compute_topic_frames(segments):
    frames = defaultdict(int)
    for seg in segments:
        topic = seg["topic"]
        duration = seg["end"] - seg["start"] + 1
        frames[topic] += duration
    return frames

# ---------------------------------------------------------
# Process each cleaned OCR file
# ---------------------------------------------------------
for file in os.listdir(input_dir):
    if not file.endswith("_clean.pkl"):
        continue

    df = pd.read_pickle(os.path.join(input_dir, file))
    newscast_name = file.replace("_clean.pkl", "")

    # Folder for TXT reports
    newscast_dir = os.path.join(output_root, newscast_name)
    os.makedirs(newscast_dir, exist_ok=True)

    print(f"\nProcessing {newscast_name}...")

    channel = extract_channel(newscast_name)
    date = extract_date(newscast_name)

    df["topic"] = df["clean_text"].apply(classify_topic)

    segments = []
    current_topic = None
    start_frame = 0

    for i, topic in enumerate(df["topic"]):
        if topic != current_topic:
            if current_topic is not None:
                segments.append({
                    "topic": current_topic,
                    "start": start_frame,
                    "end": i - 1
                })
            current_topic = topic
            start_frame = i

    segments.append({
        "topic": current_topic,
        "start": start_frame,
        "end": len(df) - 1
    })

    segments = smooth_segments_light(segments)

    seg_df = pd.DataFrame(segments)
    seg_df["duration_frames"] = seg_df["end"] - seg_df["start"] + 1

    block_count = count_topic_blocks(segments)
    frame_count = compute_topic_frames(segments)
    total_frames = sum(frame_count.values())

    # ---------------------------------------------------------
    # SAVE TXT (inside each telejornal folder)
    # ---------------------------------------------------------
    txt_path = os.path.join(newscast_dir, f"{newscast_name}_topic_blocks.txt")

    with open(txt_path, "w") as f:

        f.write("\n==============================\n")
        f.write(f"Topic Block Report: {newscast_name}\n")
        f.write("==============================\n\n")

        f.write(f"Total segments after smoothing: {len(seg_df)}\n")
        f.write(f"Total frames: {total_frames}\n\n")

        f.write("Segments:\n")
        for _, row in seg_df.iterrows():
            f.write(
                f"- {row['topic']} | frames {row['start']}–{row['end']} "
                f"({row['duration_frames']} frames)\n"
            )

        f.write("\nTopic block count (each block = 1 vote):\n")
        for topic, count in block_count.items():
            f.write(f"  {topic}: {count}\n")

        f.write("\nTotal frames per topic:\n")
        for topic, frames in frame_count.items():
            pct = (frames / total_frames) * 100
            f.write(f"  {topic}: {frames} frames ({pct:.1f}%)\n")

    print(f"Saved TXT: {txt_path}")

    # ---------------------------------------------------------
    # SAVE FINAL JSON (in root folder)
    # ---------------------------------------------------------
    final_json = {
        "newscast": newscast_name,
        "channel": channel,
        "date": date,
        "total_frames": total_frames,
        "topics": {}
    }

    for topic in frame_count:
        frames = frame_count[topic]
        pct = (frames / total_frames) * 100
        blocks = block_count.get(topic, 0)

        final_json["topics"][topic] = {
            "frames": frames,
            "percent": round(pct, 2),
            "blocks": blocks
        }

    json_path = os.path.join(output_root, f"{newscast_name}_final.json")
    with open(json_path, "w") as f:
        json.dump(final_json, f, indent=4)

    print(f"Saved FINAL JSON: {json_path}")

print("Done.")
