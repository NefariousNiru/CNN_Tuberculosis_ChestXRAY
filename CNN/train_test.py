from util import time_util
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, val_loader, optimizer, criterion, device, class_names, model_name, epochs=20):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    for epoch in range(epochs):
        start_time = time_util.get_start_time()
        print(f"Epoch [{epoch + 1}/{epochs}]")
        time_util.print_time(start_time, "started", "Epoch")

        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Convert logits to predicted class indices
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Calculate training accuracy and average loss for the epoch
        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validate the model
        val_loss, val_acc = test_model(model, val_loader, criterion, device, class_names)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        end_time = time_util.get_end_time()
        epoch_time = time_util.get_time_difference(start_time, end_time)

        print(f'Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'Time: {epoch_time:.2f} seconds')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/{model_name}')
            print("Model saved!")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


def test_model(model, val_loader, criterion, device, class_names):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()

            # Forward pass
            outputs = model(images)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Get predicted classes
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Store probabilities for ROC AUC calculation
            all_probs.extend(outputs.softmax(dim=1).cpu().numpy())

    # Calculate average loss
    val_loss = running_loss / len(val_loader)

    # Calculate accuracy
    val_acc = correct / total

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Generate classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    print("Classification Report:\n", report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", conf_matrix)

    # Calculate ROC AUC score (if applicable)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    except ValueError:
        print("\nROC AUC Score: Not applicable for single-class predictions")

    return val_loss, val_acc


def predict(model, val_loader, criterion, device, class_names):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).long()

            # Forward pass
            outputs = model(images)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Get predicted classes
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Store probabilities for ROC AUC calculation
            all_probs.extend(outputs.softmax(dim=1).cpu().numpy())

    # Calculate average loss
    val_loss = running_loss / len(val_loader)

    # Calculate accuracy
    val_acc = correct / total

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Generate classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    print("Classification Report:\n", report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", conf_matrix)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Calculate ROC AUC score (if applicable)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    except ValueError:
        print("\nROC AUC Score: Not applicable for single-class predictions")

    return val_loss, val_acc
