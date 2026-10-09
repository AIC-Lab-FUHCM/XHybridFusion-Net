import torch.nn as nn


class MLPClassifier(nn.Module):
    """GELU MLP classifier used by the final architecture."""

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
        # [B, 768] -> [B, 1024] -> [B, 512] -> [B, num_classes]
        return self.classifier(x)
