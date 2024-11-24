import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from util import data_util, time_util
from CNN.CNN import ChestXRayClassifier
from CNN import train_test

def run_custom_cnn():
    # Start timer and initialise GPU/CPU
    start_time, device = data_util.init_time_device()

    #  Get dataset loaders and data
    train_loader, train_data, val_loader, val_data, classes = data_util.load_data()

    # Calculate class weights for imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(classes)),
        y=[y for _, y in train_loader.dataset]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = ChestXRayClassifier(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Running Training")
    train_test.train_model(model, train_loader, val_loader, optimizer, criterion, device, classes)

    print("Running Test")
    train_test.test_model(model, val_loader, criterion, device, classes)

    end_time = time_util.get_end_time()
    total_time = time_util.get_time_difference(start_time, end_time)
    time_util.print_time(end_time, "finished", "Model")
    print(f"Total Time: {total_time}")