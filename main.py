import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from util import augment
from CNN.CNN import TBClassifier
from CNN import train_test


def run_tb_classifier():
    start_time = time.time()
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    transform = augment.get_transforms()
    train_data = datasets.ImageFolder('dataset/chest_XRAY/train', transform=transform)
    val_data = datasets.ImageFolder('dataset/chest_XRAY/test', transform=transform)

    # Load training and validation data
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Calculate class weights for imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_data.targets),
        y=train_data.targets
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = TBClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_test.train_model(model, train_loader, val_loader, optimizer, criterion, device)

    train_test.evaluate_model(model, val_loader, criterion, device)

    end_time = time.time()
    print(f"Training finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

    # Calculate and print the total duration
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    run_tb_classifier()


