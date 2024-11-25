import numpy as np
from sklearn.utils import compute_class_weight
from CNN.DenseNet121WithCBAM import DenseNet121WithCBAM
from util import data_util, time_util
import torch.nn as nn
import torch.optim as optim
import torch
from CNN import train_test

def run_pretrained_model():
    # Start timer and initialise GPU/CPU
    start_time, device = data_util.init_time_device()

    #  Get dataset loaders and data
    train_loader, train_data, val_loader, val_data, classes = data_util.load_data()

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(classes)),
        y=[y for _, y in train_loader.dataset]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = DenseNet121WithCBAM().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    print("Starting training...")
    train_test.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        class_names=classes,
        model_name='pre_trained_v1.pth',
        epochs=5
    )

    # Load the best model after training
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('models/pre_trained_v1.pth'))

    # Final evaluation on the validation set
    print("Evaluating the best model...")
    train_test.predict(model, val_loader, criterion, device, classes)

    end_time = time_util.get_end_time()
    total_time = time_util.get_time_difference(start_time, end_time)
    time_util.print_time(end_time, "finished", "Model")
    print(f"Total Time: {total_time}")
