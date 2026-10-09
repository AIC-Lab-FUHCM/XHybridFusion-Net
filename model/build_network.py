import torch.nn as nn

from .modules import (
    BiFormerEncoder,
    DenseNet121Backbone,
    EfficientNetB0Backbone,
    GAMFusion,
    MLPClassifier,
)


class ProposedNetwork(nn.Module):
    """BiFormer, dual-CNN, GAM, and MLP medical image classifier."""

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
            drop_path=drop_path,
        )
        self.cnn_1 = DenseNet121Backbone(
            pretrained=pretrained,
            dropout=dropout,
            freeze_blocks=cnn_1_freeze_blocks,
        )
        self.cnn_2 = EfficientNetB0Backbone(
            pretrained=pretrained,
            dropout=dropout,
            freeze_blocks=cnn_2_freeze_blocks,
        )
        self.fusion = GAMFusion(
            channels_1=self.cnn_1.out_channels,
            channels_2=self.cnn_2.out_channels,
            dropout=dropout,
        )
        self.classifier = MLPClassifier(
            input_dim=self.fusion.output_dim,
            num_classes=num_classes,
            dropout=mlp_dropout,
        )

    def forward(self, x):
        # Input and reconstructed encoder output: [B, 3, 224, 224]
        x = self.encoder(x)

        # Parallel CNN features: [B, 1024, 7, 7] and [B, 1280, 7, 7]
        feature_1 = self.cnn_1(x)
        feature_2 = self.cnn_2(x)

        # GAM tokens: [B, 49, 768]
        fused_tokens = self.fusion(feature_1, feature_2)

        # Global average pooling over the 49 spatial tokens: [B, 768]
        fused_vector = fused_tokens.mean(dim=1)

        # Class logits: [B, num_classes]
        return self.classifier(fused_vector)


def build_network(
    num_classes,
    biformer_depth,
    patch_size,
    n_win,
    image_size,
    use_pe,
    dropout,
    drop_path,
    pretrained=True,
    topk=4,
    mlp_dropout=None,
    cnn_1_freeze_blocks=0,
    cnn_2_freeze_blocks=0,
):
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
        pretrained=pretrained,
        cnn_1_freeze_blocks=cnn_1_freeze_blocks,
        cnn_2_freeze_blocks=cnn_2_freeze_blocks,
    )
