class PersonTemp(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=512, num_classes=9):
        super(PersonTemp, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        b, n, t, c, h, w = x.shape
        x = x.view(b * n * t, c, h, w)
        with torch.no_grad():
            self.backbone.eval()
            x = self.backbone(x)
        x = x.view(b * n, t, -1)
        out, (_, _) = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)