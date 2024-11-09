import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from util import augment, load, runtime_counter
from CNN.CNN import ChestXRayClassifier
from CNN import train_test

def init_time_device():
    start_time = runtime_counter.get_start_time()
    runtime_counter.print_time(start_time, "started", "Model")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    return start_time, device

def run_chest_xray_classifier():
    # Start timer and initialise GPU/CPU
    start_time, device = init_time_device()

    #  Get dataset loaders and data
    train_loader, train_data, val_loader, val_data, classes = load.load_data()

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

    end_time = runtime_counter.get_end_time()
    total_time = runtime_counter.get_time_difference(start_time, end_time)
    runtime_counter.print_time(end_time, "finished", "Model")
    runtime_counter.print_time(total_time, "total run time", "Model")

if __name__ == "__main__":
    run_chest_xray_classifier()


