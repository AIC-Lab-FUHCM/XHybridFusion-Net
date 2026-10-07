import torch.nn as nn

from .modules import BiFormerEncoder


class ProposedNetwork(nn.Module):
    def __init__(
        self,
        num_classes,
        biformer_depth,
        patch_size,
        n_win,
        topk,
        image_size,
        use_pe,
        dropout,
        mlp_dropout,
        drop_path,
        cnn_1_class,
        cnn_2_class,
        fusion_class,
        classifier_class,
        fusion_dim=256,
        fusion_heads=8,
        pretrained=True,
        cnn_1_freeze_blocks=0,
        cnn_2_freeze_blocks=0,
    ):
        super().__init__()

        self.encoder = BiFormerEncoder(
            depth=biformer_depth,
            patch_size=patch_size,
            n_win=n_win,
            topk=topk,
            image_size=image_size,
            use_pe=use_pe,
            drop_path=drop_path
        )
        self.cnn_1 = cnn_1_class(
            pretrained=pretrained,
            dropout=dropout,
            freeze_blocks=cnn_1_freeze_blocks
        )
        self.cnn_2 = cnn_2_class(
            pretrained=pretrained,
            dropout=dropout,
            freeze_blocks=cnn_2_freeze_blocks
        )

        self.fusion = fusion_class(
            channels_1=self.cnn_1.out_channels,
            channels_2=self.cnn_2.out_channels,
            embed_dim=fusion_dim,
            num_heads=fusion_heads,
            dropout=dropout,
        )

        self.classifier = classifier_class(
            input_dim=self.fusion.output_dim,
            num_classes=num_classes,
            dropout=mlp_dropout
        )

    def forward(self, x):
        # Input: [B, 3, 224, 224]
        # BiFormer encoder: [B, 3, 224, 224]
        x = self.encoder(x)

        # Hai CNN chạy song song.
        # Với CrossAttentionFusion:
        # Hướng 1: CNN 1 (ResNet50) là Query,
        # CNN 2 (EfficientNet) là Key và Value.
        # Hướng 2: CNN 2 (EfficientNet) là Query,
        # CNN 1 (ResNet50) là Key và Value.
        feature_1 = self.cnn_1(x)  # [B, C1, H1, W1]
        feature_2 = self.cnn_2(x)  # [B, C2, H2, W2]

        # GAM mới: [B, 49, 768]. GAM cũ: [B, 49, 256].
        # Cross Attention hai chiều: [B, 49, 2 x fusion_dim].
        # Các fusion khác: [B, 49, fusion_dim].
        fused_tokens = self.fusion(feature_1, feature_2)

        # Global average trên các spatial token.
        fused_vector = fused_tokens.mean(dim=1)

        # Classification: [B, num_classes]
        logits = self.classifier(fused_vector)

        return logits


def build_network(
    num_classes,
    biformer_depth,
    patch_size,
    n_win,
    image_size,
    use_pe,
    dropout,
    drop_path,
    cnn_1_class,
    cnn_2_class,
    fusion_class,
    classifier_class,
    fusion_dim=256,
    fusion_heads=8,
    pretrained=True,
    topk=4,
    mlp_dropout=None,
    cnn_1_freeze_blocks=0,
    cnn_2_freeze_blocks=0
):
    # Main cũ không truyền mlp_dropout thì MLP dùng dropout chung như trước.
    if mlp_dropout is None:
        mlp_dropout = dropout

    return ProposedNetwork(
        num_classes=num_classes,
        biformer_depth=biformer_depth,
        patch_size=patch_size,
        n_win=n_win,
        topk=topk,
        image_size=image_size,
        use_pe=use_pe,
        dropout=dropout,
        mlp_dropout=mlp_dropout,
        drop_path=drop_path,
        cnn_1_class=cnn_1_class,
        cnn_2_class=cnn_2_class,
        fusion_class=fusion_class,
        classifier_class=classifier_class,
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
        pretrained=pretrained,
        cnn_1_freeze_blocks=cnn_1_freeze_blocks,
        cnn_2_freeze_blocks=cnn_2_freeze_blocks
    )
