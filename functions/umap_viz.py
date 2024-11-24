import numpy as np
from matplotlib import pyplot as plt
from util import data_util
import umap

def run_umap(train_feature_file, train_label_file, test_feature_file, test_label_file):
    # Preprocess and load data
    train_features, train_labels, test_features, test_labels, classes = (
        data_util.feature_pre_processing(train_feature_file, train_label_file, test_feature_file, test_label_file, smote=False))

    reducer = umap.UMAP(n_components=2, random_state=42)
    features_umap = reducer.fit_transform(train_features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], c=train_labels, cmap='viridis', alpha=0.7)
    colorbar = plt.colorbar(scatter, ticks=np.arange(len(classes)))
    colorbar.ax.set_yticklabels(classes)
    plt.title("UMAP Visualization of Feature Space (2D)")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.show()

    reducer_3d = umap.UMAP(n_components=3, random_state=42)
    features_umap_3d = reducer_3d.fit_transform(train_features)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(features_umap_3d[:, 0], features_umap_3d[:, 1], features_umap_3d[:, 2],
                         c=train_labels, cmap='viridis', alpha=0.7)
    colorbar = plt.colorbar(scatter, ticks=np.arange(len(classes)))
    colorbar.ax.set_yticklabels(classes)
    ax.set_title("UMAP Visualization of Feature Space (3D)")
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    ax.set_zlabel("UMAP Component 3")
    plt.show()
