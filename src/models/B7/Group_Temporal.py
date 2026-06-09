import torch
import torch.nn as nn

class GroupActivityB7(nn.Module):
    def __init__(self, player_model, hidden_size = 512, num_classes=8):
        super(GroupActivityB7, self).__init__()

        base = player_model.module if hasattr(player_model, 'module') else player_model

        self.resnet50 = base.backbone
        self.lstm1 = base.lstm

        for param in self.resnet50.parameters():
            param.requires_grad = False

        for param in self.lstm1.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveMaxPool2d((1, 2048))
        self.layer_norm_feat = nn.LayerNorm(2048 + hidden_size)
        self.layer_norm_pool = nn.LayerNorm(2048)

        self.lstm2 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )


    def forward(self, x):
        b, n, t, c, h, w = x.shape

        x = x.view(b * n * t, c, h, w)
        x = self.resnet50(x)

        x = x.view(b * n, t, -1)
        out , _ = self.lstm1(x)

        x = torch.cat([x, out] , dim=2)
        x = self.layer_norm_feat(x)
        x = x.contiguous()

        x = x.view(b * t, n, -1)
        x = self.pool(x)

        x = x.view(b, t, -1)
        x = self.layer_norm_pool(x)

        x, (_, _) = self.lstm2(x)
        x = x [:, -1, :]

        return self.classifier(x)



