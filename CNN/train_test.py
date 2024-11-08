import torch

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=20):
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
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f'Loss: {running_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../models/best_model_v1.pth')
            print("Model saved!")


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()

            # Forward pass
            outputs = model(images).squeeze()

            # Apply sigmoid activation if using BCEWithLogitsLoss
            predictions = torch.sigmoid(outputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Convert logits to binary predictions using a threshold of 0.5
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc
