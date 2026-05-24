import torch
import torch.nn as nn

class GroupActivityB6(nn.Module):
    def __init__(self, player_model, hidden_size = 512, num_classes=8):
        super(GroupActivityB6, self).__init__()

        base = player_model.module if hasattr(player_model, 'module') else player_model

        self.resnet50 = base.backbone

        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, n, t, c, h, w = x.shape
        x = x.view(b * n * t, c, h, w)
        x = self.resnet50(x)

        x = x.view(b * t, n, -1)
        x = self.pool(x)

        x = x.squeeze(dim=1)
        x = x.view(b, t, -1)

        x, (_, _) = self.lstm(x)
        x = x [:, -1, :]

        x = self.fc(x)
        return x



