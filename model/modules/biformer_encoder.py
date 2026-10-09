import torch
import torch.nn as nn

from third_party.BiFormer.models.biformer import Block


class BiFormerEncoder(nn.Module):
    """Apply BiFormer blocks to image patches, then reconstruct an RGB image."""

    def __init__(
        self,
        depth=4,
        num_heads=4,
        topk=4,
        patch_size=4,
        n_win=7,
        image_size=224,
        use_pe=False,
        drop_path=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Preserve every RGB patch value in the embedding dimension so the
        # feature map can be rearranged back to image resolution later.
        embed_dim = 3 * patch_size * patch_size
        if embed_dim % num_heads != 0:
            raise ValueError("The patch embedding dimension must divide num_heads")

        # [B, 3, image_size, image_size]
        # -> [B, 3 * patch_size^2, grid_size, grid_size]
        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if use_pe:
            grid_size = image_size // patch_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, grid_size, grid_size)
            )
        else:
            self.pos_embed = None

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    n_win=n_win,
                    topk=topk,
                    drop_path=drop_path,
                    # With 224 x 224 input, patch_size=4, and n_win=7,
                    # each routing region is 8 x 8. Average pooling reduces
                    # K/V to 4 x 4 through a deterministic CUDA backward path.
                    kv_per_win=4,
                    kv_downsample_mode="avgpool",
                    kv_downsample_ratio=2,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # Default shapes: [B, 3, 224, 224] -> [B, 48, 56, 56]
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Every BiFormer block preserves [B, 48, 56, 56].
        x = self.blocks(x)
        x = self.norm(x)

        # Pixel-shuffle-style rearrangement reverses the spatial packing made
        # by patch embedding. It does not invert the learned Conv2d weights.
        batch_size, _, height, width = x.shape
        x = x.reshape(
            batch_size,
            3,
            self.patch_size,
            self.patch_size,
            height,
            width,
        )
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(
            batch_size,
            3,
            height * self.patch_size,
            width * self.patch_size,
        )

        # [B, 48, 56, 56] -> [B, 3, 224, 224]
        return x
