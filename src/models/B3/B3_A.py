import torch.nn as nn
import torch
import torchvision.models as models

class ResNetFineTune(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(l) for l in ['layer1', 'layer2']):
                param.requires_grad = False

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)