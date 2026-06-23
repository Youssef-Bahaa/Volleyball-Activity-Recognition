import torch
import torch.nn as nn
import torchvision.models as models

class PersonTemp(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=512, num_classes=9):
        super(PersonTemp, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        b, n, t, c, h, w = x.shape

        x = x.view(b * n * t, c, h, w)
        x = self.backbone(x)          # [b*n*t, 2048, 1, 1]

        x = x.view(b * n, t, -1)     # [b*n, t, 2048]
        out, _ = self.lstm(x)
        out = out[:, -1, :]           # [b*n, 512]

        return self.fc(out)