import torch
import torch.nn as nn

class GroupActivityB5(nn.Module):
    def __init__(self, player_model, num_classes=8):
        super(GroupActivityB5, self).__init__()

        self.resnet50 = player_model.backbone
        self.lstm = player_model.lstm

        for param in self.resnet50.parameters():
            param.requires_grad = False

        for param in self.lstm.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveMaxPooling((1, 2048))

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Relu(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, n, t, c, h, w = x.shape
        x = x.view(b * n * t, c, h, w)
        x = self.resnet50(x)

        x = x.view(b * n, t, -1)
        out, (h, c) = self.lstm(x)

        x = torch.cat([x, out], dim=2)
        x = x.contiguous()
        x = x[:, -1, :]

        x = x.view(b, n, -1)
        x = self.pool(x)
        x = x.squeeze(dim=1)

        return self.fc(x)