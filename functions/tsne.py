import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from util import data_util

def run_tsne(train_feature_file, train_label_file, test_feature_file, test_label_file):
    # Preprocess and load data
    train_features, train_labels, test_features, test_labels, classes = (
        data_util.feature_pre_processing(train_feature_file, train_label_file, test_feature_file, test_label_file, smote=False))

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    train_features_tsne = tsne.fit_transform(train_features)

    # Scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=train_labels, cmap='viridis',
                          alpha=0.7)
    colorbar = plt.colorbar(scatter, ticks=np.arange(len(classes)))
    colorbar.ax.set_yticklabels(classes)
    plt.title("t-SNE Visualization of Feature Space (2D)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
    train_features_tsne_3d = tsne_3d.fit_transform(train_features)

    # 3D Scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(train_features_tsne_3d[:, 0], train_features_tsne_3d[:, 1], train_features_tsne_3d[:, 2],
                         c=train_labels, cmap='viridis', alpha=0.7)
    colorbar = plt.colorbar(scatter, ticks=np.arange(len(classes)))
    colorbar.ax.set_yticklabels(classes)
    ax.set_title("t-SNE Visualization of Feature Space (3D)")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_zlabel("t-SNE Component 3")
    plt.show()
