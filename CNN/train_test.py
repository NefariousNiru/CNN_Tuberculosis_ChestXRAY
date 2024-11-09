import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def train_model(model, train_loader, val_loader, optimizer, criterion, device, class_names, epochs=20):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            # Forward pass
            outputs = model(images).squeeze()

            # Apply sigmoid activation if using BCEWithLogitsLoss
            predictions = torch.sigmoid(outputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Convert logits to binary predictions using a threshold of 0.5
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_loss, val_acc = test_model(model, val_loader, criterion, device, class_names)

        print(f'Loss: {running_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model_v1.pth')
            print("Model saved!")


def test_model(model, val_loader, criterion, device, class_names):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Get predicted classes
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate average loss
    val_loss = running_loss / len(val_loader)

    # Calculate accuracy
    val_acc = correct / total

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Generate classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    print("\nClassification Report:\n", report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Calculate ROC AUC score (if applicable)
    try:
        roc_auc = roc_auc_score(all_labels, outputs.softmax(dim=1).cpu().numpy(), multi_class='ovr')
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    except ValueError:
        print("\nROC AUC Score: Not applicable for single-class predictions")

    return val_loss, val_acc
