import torch.nn as nn


class MLPGELUClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # GAM: [B, 768] -> [B, 1024] -> [B, 512] -> [B, num_classes].
        return self.classifier(x)


class MLPReLUClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # GAM: [B, 768] -> [B, 1024] -> [B, 512] -> [B, num_classes].
        return self.classifier(x)


class MLPOldClassifier(nn.Module):
    """MLP cũ: input -> input/2 -> số lớp, activation ReLU."""

    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
