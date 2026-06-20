import torch
import torch.nn as nn
import torchvision.models as models

class PersonTemp(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=512, num_classes=9):
        super(PersonTemp, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.layer_norm = nn.LayerNorm(input_dim)

        for i, child in enumerate(self.backbone.children()):
            requires_grad = i >= 6
            for param in child.parameters():
                param.requires_grad = requires_grad

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes),
        )


    def forward(self, x):
        b, n, t, c, h, w = x.shape

        x = x.view(b * n * t, c, h, w)
        x = self.backbone(x)

        x = x.view(b * n, t, -1)
        x = self.layer_norm(x)

        out, (_, _) = self.lstm(x)
        out = out[:, -1, :]

        return self.fc(out)