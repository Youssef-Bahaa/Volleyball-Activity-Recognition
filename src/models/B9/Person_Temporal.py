import torch
import torch.nn as nn
import torchvision.models as models


class TemporalAttention(nn.Module):
    """
    Learns which frame matters most for classifying each player's action.
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        # x: [B, T, dim]
        w = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        return (x * w).sum(dim=1)               # [B, dim]


class PersonTemp(nn.Module):
    def __init__(self, hidden_size=512, num_classes=9):
        super(PersonTemp, self).__init__()

        self.feature_dim = 768  # ConvNeXt-Tiny output dim

        convnext = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        self.backbone = nn.Sequential(
            convnext.features,   # spatial feature extraction
            convnext.avgpool,    # global average pooling
            nn.Flatten(1)        # (B, 768)
        )

        # Freeze early stages, finetune last 2
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in convnext.features[6].parameters():  # stage 3
            param.requires_grad = True
        for param in convnext.features[7].parameters():  # stage 4
            param.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        lstm_dim = hidden_size * 2  # bidirectional doubles output

        # Temporal attention — replaces out[:, -1, :]
        self.temporal_attn = TemporalAttention(lstm_dim)

        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, N, T, C, H, W)
        b, n, t, c, h, w = x.shape

        # Extract ConvNeXt features per player per frame
        x = x.view(b * n * t, c, h, w)
        x = self.backbone(x)               # [b*n*t, 768]

        # Temporal modeling per player
        x = x.view(b * n, t, -1)          # [b*n, t, 768]
        out, _ = self.lstm(x)             # [b*n, t, hidden*2]

        # Attend to most important frame
        out = self.temporal_attn(out)     # [b*n, hidden*2]

        return self.fc(out)               # [b*n, num_classes]