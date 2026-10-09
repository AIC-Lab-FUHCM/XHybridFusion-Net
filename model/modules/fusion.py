import torch
import torch.nn as nn


class GAMFusion(nn.Module):
    """Fuse two CNN feature maps with channel and spatial attention."""

    output_dim = 768

    def __init__(self, channels_1, channels_2, dropout=0.1, reduction=4):
        super().__init__()
        input_channels = channels_1 + channels_2
        reduced_channels = input_channels // reduction

        self.channel_attention = nn.Sequential(
            nn.Linear(input_channels, reduced_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_channels, input_channels),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(input_channels, reduced_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(reduced_channels, input_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(input_channels),
            nn.Sigmoid(),
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(input_channels, self.output_dim, kernel_size=1),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(),
        )

    def forward(self, feature_1, feature_2):
        # [B, 1024, 7, 7] + [B, 1280, 7, 7] -> [B, 2304, 7, 7]
        x = torch.cat([feature_1, feature_2], dim=1)

        # Channel attention assigns one weight per channel and spatial location.
        channel_weights = x.permute(0, 2, 3, 1)
        channel_weights = self.channel_attention(channel_weights)
        channel_weights = channel_weights.permute(0, 3, 1, 2)
        x = x * channel_weights

        # Spatial attention refines the attended feature map.
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights

        # [B, 2304, 7, 7] -> [B, 768, 7, 7] -> [B, 49, 768]
        x = self.output_projection(x)
        return x.flatten(2).transpose(1, 2)
