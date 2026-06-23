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

        # pools over 6 players, keeps 2048 features -> squeeze -> [b*t, 2048]
        self.pooling = nn.AdaptiveMaxPool2d((1, 2048))

        self.lstm2 = nn.LSTM(
            input_size=4096,   # cat(left_team, right_team) = 2048 + 2048
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
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
        x = self.resnet50(x)               # [b*n*t, 2048, 1, 1]

        x = x.view(b * n, t, -1)          # [b*n, t, 2048]
        out, _ = self.lstm1(x)             # [b*n, t, 512]

        x = torch.cat([x, out], dim=-1)    # [b*n, t, 2560]
        x = x.view(b, n, t, -1)           # [b, n, t, 2560]
        x = x.permute(0, 2, 1, 3)         # [b, t, n, 2560]
        x = x.contiguous().view(b*t, n, -1)  # [b*t, n, 2560]

        left_team = x[:, :6, :]           # [b*t, 6, 2560]
        right_team = x[:, 6:, :]           # [b*t, 6, 2560]

        # pool over 6 players → [b*t, 1, 2048] -> squeeze -> [b*t, 2048]
        left_team  = self.pooling(left_team).squeeze(1)
        right_team = self.pooling(right_team).squeeze(1)

        x = torch.cat([left_team, right_team], dim=1)  # [b*t, 4096]
        x = x.view(b, t, -1)              # [b, t, 4096]

        x, _ = self.lstm2(x)              # [b, t, 512]
        x = x[:, -1, :]                   # [b, 512]

        return self.fc(x)