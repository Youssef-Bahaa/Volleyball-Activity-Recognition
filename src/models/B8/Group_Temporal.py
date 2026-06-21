import torch
import torch.nn as nn

class GroupActivityB8(nn.Module):
    def __init__(self, player_model, hidden_size=512, num_classes=8):
        super(GroupActivityB8, self).__init__()

        base = player_model.module if hasattr(player_model, 'module') else player_model

        self.resnet50 = base.backbone
        self.lstm1 = base.lstm

        for param in self.resnet50.parameters():
            param.requires_grad = False
        for param in self.lstm1.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveMaxPool2d((1, 1024))
        self.layer_norm = nn.LayerNorm(2048)

        self.lstm2 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        b, n, t, c, h, w = x.shape
        x = x.view(b * n * t, c, h, w)
        x = self.resnet50(x)

        x = x.view(b * n, t, -1)
        x = self.layer_norm(x)
        out, _ = self.lstm1(x)

        x = torch.cat([x, out], dim=2).contiguous()

        x = x.view(b * t, n, -1)
        team1 = x[:, :6, :]
        team2 = x[:, 6:, :]

        team1 = self.pool(team1)
        team2 = self.pool(team2)

        x = torch.cat([team1, team2], dim=1)

        x = x.view(b, t, -1)
        x = self.layer_norm(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]

        return self.classifier(x)