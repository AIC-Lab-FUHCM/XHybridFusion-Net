import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    densenet121,
    efficientnet_b0,
)


class DenseNet121Backbone(nn.Module):
    """DenseNet121 feature extractor that returns a 7 x 7 feature map."""

    out_channels = 1024

    def __init__(self, pretrained=True, dropout=0.1, freeze_blocks=0):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        self.features = nn.Sequential(model.features, nn.ReLU())
        self.dropout = nn.Dropout2d(dropout)

        blocks = [
            model.features.denseblock1,
            model.features.denseblock2,
            model.features.denseblock3,
            model.features.denseblock4,
        ]
        if not 0 <= freeze_blocks <= len(blocks):
            raise ValueError("DenseNet121 freeze_blocks must be between 0 and 4")
        for block in blocks[:freeze_blocks]:
            for parameter in block.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        features = self.features[0]

        # Stem.
        x = features.conv0(x)
        x = features.norm0(x)
        x = features.relu0(x)
        x = features.pool0(x)
        x = self.dropout(x)

        # Apply dropout after each dense block and transition.
        x = features.denseblock1(x)
        x = self.dropout(x)
        x = features.transition1(x)
        x = self.dropout(x)
        x = features.denseblock2(x)
        x = self.dropout(x)
        x = features.transition2(x)
        x = self.dropout(x)
        x = features.denseblock3(x)
        x = self.dropout(x)
        x = features.transition3(x)
        x = self.dropout(x)
        x = features.denseblock4(x)
        x = self.dropout(x)
        x = features.norm5(x)
        x = self.features[1](x)
        x = self.dropout(x)

        # [B, 3, 224, 224] -> [B, 1024, 7, 7]
        return x


class EfficientNetB0Backbone(nn.Module):
    """EfficientNet-B0 feature extractor that returns a 7 x 7 feature map."""

    out_channels = 1280

    def __init__(self, pretrained=True, dropout=0.1, freeze_blocks=0):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        self.features = model.features
        self.dropout = nn.Dropout2d(dropout)

        blocks = list(self.features.children())
        if not 0 <= freeze_blocks <= len(blocks):
            raise ValueError(
                f"EfficientNet-B0 freeze_blocks must be between 0 and {len(blocks)}"
            )
        for block in blocks[:freeze_blocks]:
            for parameter in block.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        # Apply dropout after each convolutional or MBConv stage.
        for layer in self.features:
            x = layer(x)
            x = self.dropout(x)

        # [B, 3, 224, 224] -> [B, 1280, 7, 7]
        return x
