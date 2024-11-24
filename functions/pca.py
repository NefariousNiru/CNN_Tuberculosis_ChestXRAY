from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from util import data_util
import numpy as np

def run_pca(train_feature_file, train_label_file, test_feature_file, test_label_file):
    # Preprocess and load data
    train_features, train_labels, test_features, test_labels, classes = (
        data_util.feature_pre_processing(train_feature_file, train_label_file, test_feature_file, test_label_file, smote=False))

    # PCA 2D Visualization
    print("PCA 2-D")
    pca_2d = PCA(n_components=2)
    train_features_pca_2d = pca_2d.fit_transform(train_features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(train_features_pca_2d[:, 0], train_features_pca_2d[:, 1], c=train_labels, cmap='viridis', alpha=0.7)
    colorbar = plt.colorbar(scatter, ticks=np.arange(len(classes)))
    colorbar.ax.set_yticklabels(classes)  # Map colorbar to class names
    plt.title("PCA Visualization of Feature Space (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    explained_variance = pca_2d.explained_variance_ratio_
    print(f"Explained Variance by Principal Components: {explained_variance}")

    # PCA 3D Visualization
    print("\nPCA 3-D")
    pca_3d = PCA(n_components=3)
    train_features_pca_3d = pca_3d.fit_transform(train_features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(train_features_pca_3d[:, 0], train_features_pca_3d[:, 1], train_features_pca_3d[:, 2],
                          c=train_labels, cmap='viridis', alpha=0.7)
    colorbar = plt.colorbar(scatter, ticks=np.arange(len(classes)))
    colorbar.ax.set_yticklabels(classes)  # Map colorbar to class names
    ax.set_title("PCA Visualization of Feature Space (3D)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.show()

    explained_variance_3d = pca_3d.explained_variance_ratio_
    print(f"Explained Variance by Principal Components (3D): {explained_variance_3d}")

    cumulative_variance = np.cumsum(PCA().fit(train_features).explained_variance_ratio_)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--',
             label='Cumulative Variance')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")

    # Set custom y-ticks for finer granularity
    plt.yticks(np.linspace(0, 1, 21))
    plt.xticks(np.linspace(0, 1000, 11))
    # Add grid and show the plot
    plt.grid()
    plt.legend()
    plt.show()
