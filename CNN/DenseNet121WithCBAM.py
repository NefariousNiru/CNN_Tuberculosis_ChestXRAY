from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Remove the classification layer to implement others
        self.base_model.classifier = nn.Identity()
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x
