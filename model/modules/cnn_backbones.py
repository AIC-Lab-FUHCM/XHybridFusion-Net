import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
    efficientnet_b1,
    resnet18,
    resnet50,
)


class ResNet18Backbone(nn.Module):
    out_channels = 512

    def __init__(self, pretrained=True, dropout=0.1, freeze_blocks=0):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # Stem.
        for layer in self.features[:4]:
            x = layer(x)
        x = self.dropout(x)

        # Dropout sau từng residual stage: layer1, layer2, layer3, layer4.
        for layer in self.features[4:]:
            x = layer(x)
            x = self.dropout(x)

        # [B, 3, 224, 224] -> [B, 512, 7, 7]
        return x


class ResNet50Backbone(nn.Module):
    out_channels = 2048

    def __init__(self, pretrained=True, dropout=0.1, freeze_blocks=0):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # Stem.
        for layer in self.features[:4]:
            x = layer(x)
        x = self.dropout(x)

        # Dropout sau từng residual stage: layer1, layer2, layer3, layer4.
        for layer in self.features[4:]:
            x = layer(x)
            x = self.dropout(x)

        # [B, 3, 224, 224] -> [B, 2048, 7, 7]
        return x


class EfficientNetB0Backbone(nn.Module):
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
                f"freeze_blocks của EfficientNetB0 phải từ 0 đến {len(blocks)}"
            )
        for block in blocks[:freeze_blocks]:
            for parameter in block.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        # Dropout sau từng Conv/MBConv stage của EfficientNet.
        for layer in self.features:
            x = layer(x)
            x = self.dropout(x)

        # [B, 3, 224, 224] -> [B, 1280, 7, 7]
        return x


class EfficientNetB1Backbone(nn.Module):
    out_channels = 1280

    def __init__(self, pretrained=True, dropout=0.1, freeze_blocks=0):
        super().__init__()
        weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
        model = efficientnet_b1(weights=weights)
        self.features = model.features
        self.dropout = nn.Dropout2d(dropout)

        blocks = list(self.features.children())
        if not 0 <= freeze_blocks <= len(blocks):
            raise ValueError(
                f"freeze_blocks của EfficientNetB1 phải từ 0 đến {len(blocks)}"
            )
        for block in blocks[:freeze_blocks]:
            for parameter in block.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        # Dropout sau từng Conv/MBConv stage của EfficientNet.
        for layer in self.features:
            x = layer(x)
            x = self.dropout(x)

        # [B, 3, 224, 224] -> [B, 1280, 7, 7]
        return x


class DenseNet121Backbone(nn.Module):
    out_channels = 1024

    def __init__(self, pretrained=True, dropout=0.1, freeze_blocks=0):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = densenet121(weights=weights)
        self.features = nn.Sequential(
            model.features,
            nn.ReLU()
        )
        self.dropout = nn.Dropout2d(dropout)

        blocks = [
            model.features.denseblock1,
            model.features.denseblock2,
            model.features.denseblock3,
            model.features.denseblock4,
        ]
        if not 0 <= freeze_blocks <= len(blocks):
            raise ValueError("freeze_blocks của DenseNet121 phải từ 0 đến 4")
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

        # Dropout sau từng DenseBlock và Transition.
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
