import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import compute_class_weight

from CNN import train_test
from CNN.CNN import ChestXRayClassifier
from CNN.ExtendedCNN import ExtendedChestXRayClassifier
from util import data_util

def run_fine_tune_customCNN(path):
    base_model = ChestXRayClassifier()
    base_model.load_state_dict(torch.load(path))

    _, device = data_util.init_time_device()

    train_loader, train_data, val_loader, val_data, classes = data_util.load_data()

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(classes)),
        y=[y for _, y in train_loader.dataset]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    fine_tune = ExtendedChestXRayClassifier(base_model).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(fine_tune.parameters(), lr=0.0001)

    train_test.train_model(fine_tune, train_loader, val_loader, optimizer, criterion, device, classes,
                           'fine_tuned_custom_model_v1.pth', epochs=10)

    # Load the best model after training
    print("Loading best model for final evaluation...")
    fine_tune.load_state_dict(torch.load('models/fine_tuned_custom_model_v1.pth'))

    print("Running Predictions on test set")
    train_test.predict(fine_tune, val_loader, criterion, device, classes)
