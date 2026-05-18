import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


folder = "Project_Features"
target = "Marques_Mendes_vs_Gouveia_Melo_December_21"
output_folder = "part2_v03"
visual_df, audio_df = None, None

# =====================================================================
# Check Files Formats
# =====================================================================
for file in os.listdir(folder):
    path = os.path.join(folder, file)
    if file.endswith("_visual.pkl") and target in file:
        visual_df = pd.read_pickle(path)
        print(f"\n=== VISUAL FILE ===\n{file}\n")
        visual_df.info()
        sample_face = visual_df["Fer"].dropna().iloc[0]
        first_face = sample_face[0]
        sample_pose = visual_df["Poses"].dropna().iloc[0]
        first_pose = sample_pose[0]
        print("\n--- FACE STRUCTURE ---")
        print("Detections in sample frame:", len(sample_face))
        print("Keys:", first_face.keys())
        print("\n--- POSE STRUCTURE ---")
        print("Detections in sample frame:", len(sample_pose))
        print("Keys:", first_pose.keys())
    if file.endswith("_audio.pkl") and target in file:
        audio_df = pd.read_pickle(path)
        print(f"\n=== AUDIO FILE ===\n{file}\n")
        audio_df.info()


# =====================================================================
# COLLECT POSE KEYPOINTS - PCA and TSNE
# =====================================================================
X = []  
X2 = []
for pose_list in visual_df["Poses"]:
    if not isinstance(pose_list, list):
        continue
    for pose in pose_list:
        if not (isinstance(pose, dict) and "pose" in pose and "bbox" in pose):
            continue
        if pose["class_conf"] < 0.5: ## 
            continue
        keypoints = pose["pose"]
        x1,y1,x2,y2 = pose["bbox"]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        if len(keypoints) != 17:
            continue
        vec = []
        vec2 = []
        for point in keypoints:
            if len(point) < 3:
                continue
            x, y, conf = point
            vec.extend([x, y]) ## , conf
            vec2.extend([x - x_center, y - y_center]) ## 
        X.append(vec)
        X2.append(vec2)
X = np.array(X)
X2 = np.array(X2)

# ----- POSE PCA X -----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(6,6))
plt.scatter(X_pca[:,0], X_pca[:,1], s=2, alpha=0.3)
plt.title("PCA of Pose Keypoints (17x3 per person)")
plt.xlabel("PC1")
plt.ylabel("PC2")
save_path = os.path.join(output_folder, f"pose_pca_X_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()

# ----- POSE PCA X2 -----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(6,6))
plt.scatter(X_pca[:,0], X_pca[:,1], s=2, alpha=0.3)
plt.title("PCA of Pose Keypoints (17x3 per person)")
plt.xlabel("PC1")
plt.ylabel("PC2")
save_path = os.path.join(output_folder, f"pose_pca_X2_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()

# ----- POSE TSNE X2 -----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X2)
tsne = TSNE(n_components=2, perplexity=30,random_state=42, init="pca")
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(6,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], s=2, alpha=0.3)
plt.title("t-SNE of Pose Keypoints (17x3 per person)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
save_path = os.path.join(output_folder, f"pose_tsne_X2_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()


# =====================================================================
# COLLECT FACE LANDMARKS - PCA and TSNE
# =====================================================================
X = []
X2 = []
y_emotion = []
for fer_list in visual_df["Fer"]:
    if not isinstance(fer_list, list):
        continue
    for face in fer_list:
        if not (isinstance(face, dict) and "landmarks" in face):
            continue
        landmarks = face["landmarks"]
        x1,y1,x2,y2 = face["bbox"]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        if len(landmarks) != 106:
            continue
        vec = []
        vec2 = []
        for point in landmarks:
            x, y = point[:2]
            vec.extend([x, y])
            vec2.extend([x-x_center, y-y_center])
        emotion = face.get("top_emotion", None)
        X.append(vec)
        X2.append(vec2)
        y_emotion.append(emotion)
X = np.array(X)
X2 = np.array(X2)
y_emotion = np.array(y_emotion)
emotions, y_idx = np.unique(y_emotion, return_inverse=True)
cmap = plt.colormaps["tab10"]
print("Face PCA matrix shape:", X.shape)

# ----- FACE PCA X -----
scaler = StandardScaler()
X_face_scaled = scaler.fit_transform(X)
pca_face = PCA(n_components=2)
X_face_pca = pca_face.fit_transform(X_face_scaled)
plt.figure(figsize=(6,6))
scatter = plt.scatter(X_face_pca[:,0], X_face_pca[:,1],c=y_idx, s=2, alpha=0.3)
handles = []
for i, emo in enumerate(emotions):
    handles.append(
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=emo,
            markerfacecolor=cmap(i),
            markersize=6
        )
    )
plt.legend(handles=handles, title="Emotion")
plt.title("PCA of Face Landmarks (106x2 per detection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
save_path = os.path.join(output_folder, f"face_pca_X_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()

# ----- FACE PCA X2 -----
scaler = StandardScaler()
X_face_scaled = scaler.fit_transform(X2)
pca_face = PCA(n_components=2)
X_face_pca = pca_face.fit_transform(X_face_scaled)
plt.figure(figsize=(6,6))
scatter = plt.scatter(X_face_pca[:,0], X_face_pca[:,1],c=y_idx, s=2, alpha=0.3)
handles = []
for i, emo in enumerate(emotions):
    handles.append(
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=emo,
            markerfacecolor=cmap(i),
            markersize=6
        )
    )
plt.legend(handles=handles, title="Emotion")
plt.title("PCA of Face Landmarks (106x2 per detection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
save_path = os.path.join(output_folder, f"face_pca_X2_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()

# ----- FACE TSNE X2 -----
scaler = StandardScaler()
X_face_scaled = scaler.fit_transform(X2)
tsne_face = TSNE( n_components=2, perplexity=30, random_state=42,init="pca")
X_face_tsne = tsne_face.fit_transform(X_face_scaled)
plt.figure(figsize=(6,6))
handles = []
scatter = plt.scatter(X_face_tsne[:,0], X_face_tsne[:,1], c=y_idx, s=2, alpha=0.3)
for i, emo in enumerate(emotions):
    handles.append(
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=emo,
            markerfacecolor=cmap(i),
            markersize=6
        )
    )
plt.legend(handles=handles, title="Emotion")
plt.title("t-SNE of Face Landmarks (106x2 centered)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
save_path = os.path.join(output_folder, f"face_tsne_X2_{target}.png")
plt.savefig(save_path, dpi=300)
plt.show()