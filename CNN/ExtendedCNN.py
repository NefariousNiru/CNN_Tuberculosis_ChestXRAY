import torch.nn as nn

class ExtendedChestXRayClassifier(nn.Module):
    def __init__(self, base_model, num_classes=4):
        super(ExtendedChestXRayClassifier, self).__init__()
        for param in base_model.parameters():
            param.requires_grad = False

        self.base_model = base_model

        self.fine_tune = nn.Sequential(
            nn.Linear(64, 512),  # First fine-tune layer with same size as input
            nn.BatchNorm1d(512),  # Batch normalization for regularization
            nn.ReLU(),  # Activation function
            nn.Dropout(0.4),  # Increased dropout to prevent overfitting

            nn.Linear(512, 256),  # Second layer reducing dimensionality
            nn.BatchNorm1d(256),  # Batch normalization
            nn.GELU(),
            nn.Dropout(0.4),  # Dropout to further prevent overfitting

            nn.Linear(256, 64),  # Third layer reducing dimensionality further
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),

            nn.Linear(64, num_classes)  # Final layer mapping to output classes
        )

    def forward(self, x):
        x = self.base_model.conv_layers(x)  # Through convolutional layers
        x = self.base_model.fc_layers[:-1](x)  # Up to the penultimate FC layer
        x = self.fine_tune(x)  # Pass through fine-tuning layers
        return x