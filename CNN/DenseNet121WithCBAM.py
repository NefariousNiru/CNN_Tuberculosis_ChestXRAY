from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn
import torch

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply channel attention
        ca = self.channel_attention(x) * x
        # Apply spatial attention
        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out, _ = torch.max(ca, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        return sa * ca

class DenseNet121WithCBAM(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseNet121WithCBAM, self).__init__()
        self.base_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.features = nn.Sequential(
            self.base_model.features,
            CBAM(channels=1024)  # Apply CBAM after DenseNet features
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        if return_features:
            return x

        x = self.classifier(x)
        return x
