import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFineTune(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        in_dim = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)