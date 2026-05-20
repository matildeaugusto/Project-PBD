# TODO: Concatenate face and pose by concatenating PCA_30's of each (do variance plot)
# TODO: Do some clustering:
    # hierarchical, k-means and variants (fuzzy,median,center,weighted), dbscan, spectral, gmm,...
    # Maybe try spectral (from scikit drawings) or fuzzy c-means
    # dendograms, elbow methods, silluette score,...

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px


def save_plotly_scatter(points,frames,title,save_html,labels=None,search_term=["047","058","525", "545","1545","1271","289","137","272"]): 
    # search_term=None 
    # 545 and 525 = MM 
    # 1545 and 1271 = Melo   
    # 047 and 058 = Moderator
    # 289 and 137 = both candidates
    # 272 = All 3 people
    # "111", "151", "153"? both candidates
    df_plot = pd.DataFrame({ "x": points[:, 0],"y": points[:, 1],"frame": frames})
    df_plot["frame_str"] = df_plot["frame"].astype(str)
    fig = px.scatter(df_plot, x="x", y="y",color=labels if labels is not None else None,hover_data=["frame_str"],  title=title, opacity=0.4)
    fig.update_traces(marker=dict(size=4))
    df_plot["frame_id"] = (df_plot["frame_str"].str.extract(r"_(\d+)\.jpg")[0])
    if search_term is not None:
        if isinstance(search_term, str):
            search_term = [search_term]
        mask = df_plot["frame_id"].isin(search_term)
        df_match = df_plot[mask]
        fig.add_scatter(x=df_match["x"], y=df_match["y"],mode="markers",name=f"Matches: {search_term}",marker=dict(size=10, color="red", symbol="circle"),hovertext=df_match["frame_str"])
    fig.write_html(save_html)
    print(f"Saved: {save_html}")


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
pose_frames = []
for frame_id, pose_list in zip(visual_df["Frame"], visual_df["Poses"]):
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
        pose_frames.append(frame_id)
X = np.array(X)
X2 = np.array(X2)

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
html_path = os.path.join(output_folder, f"pose_pca_X2_{target}.html")
save_plotly_scatter(X_pca, pose_frames,"PCA of Pose Keypoints Centered",html_path)
plt.show()

# ----- POSE PCA VARIANCE X2 -----
pca_full = PCA().fit(X_scaled)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_var) + 1), explained_var, marker="o",  label="Explained variance")
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var,marker="s",label="Cumulative variance")
plt.title("PCA Explained Variance (Pose Features)")
plt.xlabel("Number of Components")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"pose_pca_variance_X2_{target}.png"), dpi=300)
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
html_path = os.path.join(output_folder, f"pose_tsne_X2_{target}.html")
save_plotly_scatter( X_tsne, pose_frames, "t-SNE of Pose Keypoints",html_path)
plt.show()


# =====================================================================
# COLLECT FACE LANDMARKS - PCA and TSNE
# =====================================================================
X = []
X2 = []
y_emotion = []
face_frames = []
for frame_id, fer_list in zip(visual_df["Frame"], visual_df["Fer"]):
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
        face_frames.append(frame_id)
X = np.array(X)
X2 = np.array(X2)
y_emotion = np.array(y_emotion)
emotions, y_idx = np.unique(y_emotion, return_inverse=True)
cmap = plt.colormaps["tab10"]

# ----- FACE PCA X2 -----
scaler = StandardScaler()
X_face_scaled = scaler.fit_transform(X2)
pca_face = PCA(n_components=2)
X_face_pca = pca_face.fit_transform(X_face_scaled)
plt.figure(figsize=(6,6))
scatter = plt.scatter(X_face_pca[:,0], X_face_pca[:,1],c=y_idx, s=2, alpha=0.3)
handles = []
for i, emo in enumerate(emotions):
    handles.append(plt.Line2D([0], [0], marker='o',color='w',label=emo,markerfacecolor=cmap(i),markersize=6))
plt.legend(handles=handles, title="Emotion")
plt.title("PCA of Face Landmarks (106x2 per detection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
save_path = os.path.join(output_folder, f"face_pca_X2_{target}.png")
plt.savefig(save_path, dpi=300)
html_path = os.path.join(output_folder,f"face_pca_X2_{target}.html")
save_plotly_scatter(X_face_pca, face_frames, "PCA of Face Landmarks", html_path, labels=y_emotion)
plt.show()

# ----- FACE PCA VARIANCE X2 -----
pca_full = PCA().fit(X_face_scaled)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_var) + 1), explained_var, marker="o",  label="Explained variance")
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var,marker="s",label="Cumulative variance")
plt.title("PCA Explained Variance (Face Features)")
plt.xlabel("Number of Components")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"face_pca_variance_X2_{target}.png"), dpi=300)
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
    handles.append(plt.Line2D( [0], [0], marker='o', color='w',label=emo, markerfacecolor=cmap(i), markersize=6))
plt.legend(handles=handles, title="Emotion")
plt.title("t-SNE of Face Landmarks (106x2 centered)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
save_path = os.path.join(output_folder, f"face_tsne_X2_{target}.png")
plt.savefig(save_path, dpi=300)
html_path = os.path.join(output_folder,f"face_tsne_X2_{target}.html")
save_plotly_scatter( X_face_tsne, face_frames,"t-SNE of Face Landmarks", html_path, labels=y_emotion)
plt.show()
