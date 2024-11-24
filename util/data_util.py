import os
import seaborn as sns
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

from CNN.DenseNet121WithCBAM import DenseNet121WithCBAM
from util import time_util
import torch

def get_transforms():
    transform = transforms.Compose([
        # Random resizing and cropping within a safe range
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),

        # Apply random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),

        # Slight brightness and contrast adjustments
        transforms.ColorJitter(brightness=0.1, contrast=0.1),

        # Small random rotations
        transforms.RandomRotation(degrees=5),

        # Subtle affine transformations
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=3
        ),

        # Slight Gaussian blur
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),

        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Replace with dataset stats if available

        # Apply random erasing for robustness to occlusion (optional)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),

    ])
    return transform


def load_data():
    transform = get_transforms()
    train_data = datasets.ImageFolder('dataset/chest_XRAY/train', transform=transform)
    val_data = datasets.ImageFolder('dataset/chest_XRAY/test', transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    return train_loader, train_data, val_loader, val_data, train_data.classes

def save_features_from_model(features, labels, feature_file, label_file):
    np.save(feature_file, features)
    np.save(label_file, labels)
    print(f"Saved features to {feature_file} and labels to {label_file}")

def load_features_from_model(feature_file, label_file):
    if os.path.exists(feature_file) and os.path.exists(label_file):
        features = np.load(feature_file)
        labels = np.load(label_file)
        print(f"Loaded features from {feature_file} and labels from {label_file}")
        return features, labels
    else:
        print(f"Feature file {feature_file} or label file {label_file} not found.")
        return None, None


def extract_and_save_features(data_loader, model, device, feature_file, label_file):
    # Check if features already exist
    features, labels = load_features_from_model(feature_file, label_file)
    if features is not None and labels is not None:
        return features, labels

    # Extract features if not already saved
    print(f"Extracting features and labels for {feature_file}...")
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images, return_features=True)  # Forward pass through feature extractor
            outputs = outputs.view(outputs.size(0), -1).cpu().numpy()  # Flatten features
            features.append(outputs)
            labels.append(targets.numpy())

    # Combine batches and save
    features = np.vstack(features)
    labels = np.hstack(labels)
    save_features_from_model(features, labels, feature_file, label_file)

    return features, labels

def init_time_device():
    start_time = time_util.get_start_time()
    time_util.print_time(start_time, "started", "Model")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    return start_time, device

def feature_pre_processing(train_feature_file, train_label_file, test_feature_file, test_label_file, smote=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_loader, train_data, val_loader, val_data, classes = load_data()

    # Initialize Feature Extractor
    print("Initializing DenseNet121WithCBAM to extract features...")
    efficient_net = DenseNet121WithCBAM().to(device)

    # Extract Features
    print("Extracting training features...")
    train_features, train_labels = extract_and_save_features(train_loader, efficient_net, device,
                                                                       train_feature_file, train_label_file)

    print("Extracting testing features...")
    test_features, test_labels = extract_and_save_features(val_loader, efficient_net, device,
                                                                     test_feature_file, test_label_file)

    print(f"Training Features Shape: {train_features.shape}")
    print(f"Testing Features Shape: {test_features.shape}")

    if smote:
        print("Apply SMOTE")
        smote = SMOTE(random_state=42)
        train_features_balanced, train_labels_balanced = smote.fit_resample(train_features, train_labels)
        print(f"Balanced Training Features Shape: {train_features_balanced.shape}")
        print(f"Balanced Training Labels Distribution: {np.bincount(train_labels_balanced)}")
        return train_features_balanced, train_labels_balanced, test_features, test_labels, classes

    return train_features, train_labels, test_features, test_labels, classes

def f_statistics(model, features, labels, classes):
    predictions = model.predict(features)
    print(classification_report(labels, predictions, target_names=classes))

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    print(f"Confusion Matrix:\n{cm}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()