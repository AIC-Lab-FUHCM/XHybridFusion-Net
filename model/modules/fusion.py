import torch
import torch.nn as nn


class ConcatenationFusion(nn.Module):
    def __init__(
        self,
        channels_1,
        channels_2,
        embed_dim=256,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.output_dim = embed_dim
        self.projection = nn.Conv2d(channels_1 + channels_2, embed_dim, 1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, feature_1, feature_2):
        # [ CB,1, 7, 7] + [B, C2, 7, 7]
        x = torch.cat([feature_1, feature_2], dim=1)
        x = self.projection(x)
        x = self.dropout(x)
        # [B, E, 7, 7] -> [B, 49, E]
        return x.flatten(2).transpose(1, 2)


class GAMFusionNew(nn.Module):
    default_embed_dim = 768

    def __init__(
        self,
        channels_1,
        channels_2,
        embed_dim=768,
        num_heads=8,
        reduction=4,
        dropout=0.1
    ):
        super().__init__()
        self.output_dim = embed_dim
        input_channels = channels_1 + channels_2
        reduced_dim = input_channels // reduction

        # Channel attention: học mức quan trọng của từng channel.
        self.channel_attention = nn.Sequential(
            nn.Linear(input_channels, reduced_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, input_channels),
            nn.Sigmoid()
        )

        # Spatial attention: học vùng không gian quan trọng trên feature map.
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(input_channels, reduced_dim, kernel_size=7, padding=3),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(reduced_dim, input_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(input_channels),
            nn.Sigmoid()
        )

        # Sau GAM, Conv 1x1 đưa feature về 768 channel.
        self.output_projection = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

    def forward(self, feature_1, feature_2):
        # Concat: [B, C1, 7, 7] + [B, C2, 7, 7]
        #       -> [B, C1+C2, 7, 7]
        x = torch.cat([feature_1, feature_2], dim=1)

        # GAM channel attention trên feature đã concat.
        channel_weights = x.permute(0, 2, 3, 1)
        channel_weights = self.channel_attention(channel_weights)
        channel_weights = channel_weights.permute(0, 3, 1, 2)
        x = x * channel_weights

        # GAM spatial attention trên feature đã qua channel attention.
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights

        # Conv 1x1 + BatchNorm + ReLU:
        # [B, C1+C2, 7, 7] -> [B, 768, 7, 7].
        x = self.output_projection(x)

        # [B, 768, 7, 7] -> [B, 49, 768].
        return x.flatten(2).transpose(1, 2)


class GAMFusionOld(nn.Module):
    """GAM cũ: concat -> projection 1x1 -> GAM -> tokens."""

    default_embed_dim = 256

    def __init__(
        self,
        channels_1,
        channels_2,
        embed_dim=256,
        num_heads=8,
        reduction=4,
        dropout=0.1
    ):
        super().__init__()
        self.output_dim = embed_dim

        reduced_dim = embed_dim // reduction

        # Sau khi concat, đưa tổng số channel về embed_dim.
        self.projection = nn.Sequential(
            nn.Conv2d(channels_1 + channels_2, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        # Channel attention: học mức quan trọng của từng channel.
        self.channel_attention = nn.Sequential(
            nn.Linear(embed_dim, reduced_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reduced_dim, embed_dim),
            nn.Sigmoid()
        )

        # Spatial attention: học vùng không gian quan trọng trên feature map.
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(embed_dim, reduced_dim, kernel_size=7, padding=3),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(reduced_dim, embed_dim, kernel_size=7, padding=3),
            nn.BatchNorm2d(embed_dim),
            nn.Sigmoid()
        )

    def forward(self, feature_1, feature_2):
        # [B, C1, 7, 7] + [B, C2, 7, 7]
        # -> [B, C1+C2, 7, 7] -> [B, 256, 7, 7].
        x = torch.cat([feature_1, feature_2], dim=1)
        x = self.projection(x)

        # Channel attention: [B, 256, 7, 7].
        channel_weights = x.permute(0, 2, 3, 1)
        channel_weights = self.channel_attention(channel_weights)
        channel_weights = channel_weights.permute(0, 3, 1, 2)
        x = x * channel_weights

        # Spatial attention: [B, 256, 7, 7].
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights

        # [B, 256, 7, 7] -> [B, 49, 256].
        return x.flatten(2).transpose(1, 2)


# Giữ tên GAMFusion cho những file cũ; mặc định trỏ đến GAM mới.
GAMFusion = GAMFusionNew


class GatedFusion(nn.Module):
    def __init__(
        self,
        channels_1,
        channels_2,
        embed_dim=256,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.output_dim = embed_dim

        # Đưa feature của hai CNN về cùng số channel.
        self.feature_1_projection = nn.Sequential(
            nn.Conv2d(channels_1, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
            nn.Dropout2d(dropout)
        )
        self.feature_2_projection = nn.Sequential(
            nn.Conv2d(channels_2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
            nn.Dropout2d(dropout)
        )

        # Gate trong khoảng [0, 1] học mức đóng góp của từng CNN.
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.Sigmoid()
        )

    def forward(self, feature_1, feature_2):
        # [B, C1, 7, 7] -> [B, E, 7, 7]
        feature_1 = self.feature_1_projection(feature_1)

        # [B, C2, 7, 7] -> [B, E, 7, 7]
        feature_2 = self.feature_2_projection(feature_2)

        # [B, E, 7, 7] + [B, E, 7, 7] -> [B, 2E, 7, 7]
        gate = self.gate(torch.cat([feature_1, feature_2], dim=1))

        # Gate gần 1 ưu tiên CNN 1, gate gần 0 ưu tiên CNN 2.
        x = gate * feature_1 + (1.0 - gate) * feature_2

        # [B, E, 7, 7] -> [B, 49, E]
        return x.flatten(2).transpose(1, 2)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        channels_1,
        channels_2,
        embed_dim=256,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        # Hai hướng attention được concat nên output có 2 x embed_dim channel.
        self.output_dim = embed_dim * 2

        # Đưa feature của ResNet và EfficientNet về cùng embed_dim.
        self.resnet_projection = nn.Conv2d(channels_1, embed_dim, 1)
        self.efficientnet_projection = nn.Conv2d(channels_2, embed_dim, 1)

        # Hướng 1: ResNet làm Query, EfficientNet làm Key và Value.
        self.resnet_query_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Hướng 2: EfficientNet làm Query, ResNet làm Key và Value.
        self.efficientnet_query_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.activation = nn.SiLU()
        self.resnet_dropout = nn.Dropout(dropout)
        self.efficientnet_dropout = nn.Dropout(dropout)
        self.resnet_norm = nn.LayerNorm(embed_dim)
        self.efficientnet_norm = nn.LayerNorm(embed_dim)

    def forward(self, feature_1, feature_2):
        # ResNet50: [B, C1, 7, 7] -> [B, 49, E]
        resnet_tokens = self.resnet_projection(feature_1)
        resnet_tokens = resnet_tokens.flatten(2).transpose(1, 2)

        # EfficientNet: [B, C2, 7, 7] -> [B, 49, E]
        efficientnet_tokens = self.efficientnet_projection(feature_2)
        efficientnet_tokens = efficientnet_tokens.flatten(2).transpose(1, 2)

        # Hướng 1: Q = ResNet, K và V = EfficientNet.
        resnet_attention, _ = self.resnet_query_attention(
            query=resnet_tokens,
            key=efficientnet_tokens,
            value=efficientnet_tokens
        )
        resnet_attention = self.activation(resnet_attention)
        resnet_attention = self.resnet_dropout(resnet_attention)
        resnet_output = resnet_tokens + resnet_attention
        resnet_output = self.resnet_norm(resnet_output)

        # Hướng 2: Q = EfficientNet, K và V = ResNet.
        efficientnet_attention, _ = self.efficientnet_query_attention(
            query=efficientnet_tokens,
            key=resnet_tokens,
            value=resnet_tokens
        )
        efficientnet_attention = self.activation(efficientnet_attention)
        efficientnet_attention = self.efficientnet_dropout(
            efficientnet_attention
        )
        efficientnet_output = efficientnet_tokens + efficientnet_attention
        efficientnet_output = self.efficientnet_norm(efficientnet_output)

        # Concat hai hướng: [B, 49, E] + [B, 49, E] -> [B, 49, 2E].
        return torch.cat([resnet_output, efficientnet_output], dim=-1)
