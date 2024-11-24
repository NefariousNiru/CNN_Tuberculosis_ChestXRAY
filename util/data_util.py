from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from util import augment
import numpy as np

def load_data():
    transform = augment.get_transforms_v0()
    train_data = datasets.ImageFolder('dataset/chest_XRAY/train', transform=transform)
    val_data = datasets.ImageFolder('dataset/chest_XRAY/test', transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    return train_loader, train_data, val_loader, val_data, train_data.classes


def save_features_from_model(features, labels, feature_file, label_file):
    np.save(feature_file, features)
    np.save(label_file, labels)
    print(f"Saved features to {feature_file} and labels to {label_file}")