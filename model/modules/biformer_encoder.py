import torch
import torch.nn as nn

from third_party.BiFormer.models.biformer import Block


class BiFormerEncoder(nn.Module):
    def __init__(
        self,
        depth=4,
        num_heads=4,
        topk=4,
        patch_size=4,
        n_win=7,
        image_size=224,
        use_pe=False,
        drop_path=0.1
    ):
        super().__init__()
        self.patch_size = patch_size

        # Mỗi patch phải chứa đủ 3 x patch_size x patch_size giá trị
        # để sau BiFormer có thể xếp thẳng trở lại ảnh RGB.
        embed_dim = 3 * patch_size * patch_size

        if embed_dim % num_heads != 0:
            raise ValueError("Số channel sau chia patch phải chia hết cho num_heads")

        # Chia ảnh thành các patch có kích thước patch_size x patch_size.
        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # PE bổ sung sau khi chia patch. Mặc định tắt trong main.py.
        if use_pe:
            grid_size = image_size // patch_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, grid_size, grid_size)
            )
        else:
            self.pos_embed = None

        # Số block được chọn bằng BIFORMER_DEPTH trong main.py.
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    n_win=n_win,
                    topk=topk,
                    drop_path=drop_path,
                    # Với cấu hình 224, patch_size=4 và n_win=7:
                    # mỗi region là 8x8. AvgPool2d kernel/stride 2 đưa
                    # K/V về 4x4 giống adaptive pooling, nhưng backward
                    # trên CUDA hỗ trợ deterministic.
                    kv_per_win=4,
                    kv_downsample_mode="avgpool",
                    kv_downsample_ratio=2,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # Cấu hình mặc định:
        # B = batch size, ảnh = 224 x 224, patch_size = 4.

        # Ảnh RGB đầu vào.
        # [B, 3, 224, 224]
        x = self.patch_embed(x)

        # Chia ảnh thành patch 4 x 4 và chiếu lên 48 channel.
        # 48 = 3 channel RGB x 4 x 4 pixel trong mỗi patch.
        # 224 / 4 = 56.
        # [B, 3, 224, 224] -> [B, 48, 56, 56]

        # Nếu USE_PE = True, cộng positional embedding có thể học được.
        # [B, 48, 56, 56] + [1, 48, 56, 56]
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Bên trong mỗi Block của file chính thức models/biformer.py:
        #
        # 1. Depthwise Conv2D 3 x 3 tạo positional embedding:
        #    [B, 48, 56, 56] -> [B, 48, 56, 56]
        #    Sau đó cộng với input.
        #
        # 2. Đổi NCHW sang NHWC và LayerNorm:
        #    [B, 48, 56, 56] -> [B, 56, 56, 48]
        #    LayerNorm không đổi shape.
        #
        # 3. BRA chia feature map thành 7 x 7 = 49 region:
        #    [B, 56, 56, 48] -> [B, 49, 8, 8, 48]
        #
        # 4. Linear projection tạo Query, Key và Value:
        #    Q:  [B, 49, 8, 8, 48]
        #    KV: [B, 49, 8, 8, 96]
        #
        # 5. Query giữ 64 token trong mỗi region:
        #    Q: [B, 49, 64, 48]
        #    Key và Value được giảm còn 4 x 4 = 16 token mỗi region:
        #    KV: [B, 49, 16, 96]
        #
        # 6. Region routing chọn TOPK region liên quan:
        #    Routing weight: [B, 49, TOPK]
        #    Routing index:  [B, 49, TOPK]
        #
        # 7. Chia thành 4 attention head, mỗi head có 48 / 4 = 12 channel:
        #    Q: [B x 49, 4, 64, 12]
        #    K: [B x 49, 4, 12, TOPK x 16]
        #    V: [B x 49, 4, TOPK x 16, 12]
        #    Số token K/V = TOPK x 16 token.
        #
        # 8. Token attention:
        #    Attention: [B x 49, 4, 64, TOPK x 16]
        #    Output:    [B, 56, 56, 48]
        #
        # 9. Depthwise Conv2D 5 x 5 tạo LePE từ Value:
        #    [B, 48, 56, 56] -> [B, 48, 56, 56]
        #    Sau đó đổi thành [B, 56, 56, 48] và cộng vào output BRA.
        #
        # 10. Linear output và residual của BRA:
        #     [B, 56, 56, 48] -> [B, 56, 56, 48]
        #
        # 11. LayerNorm và MLP:
        #     [B, 56, 56, 48]
        #     -> [B, 56, 56, 192]
        #     -> [B, 56, 56, 48]
        #     Sau đó cộng residual.
        #
        # 12. Đổi NHWC trở lại NCHW:
        #     [B, 56, 56, 48] -> [B, 48, 56, 56]
        #
        # Vì vậy mọi Block đều giữ nguyên shape:
        # Block 1: [B, 48, 56, 56] -> [B, 48, 56, 56]
        # Block 2: [B, 48, 56, 56] -> [B, 48, 56, 56]
        # Block 3: [B, 48, 56, 56] -> [B, 48, 56, 56]
        # Block 4: [B, 48, 56, 56] -> [B, 48, 56, 56]
        x = self.blocks(x)

        # BatchNorm2D cuối không đổi shape.
        # [B, 48, 56, 56] -> [B, 48, 56, 56]
        x = self.norm(x)

        # Làm ngược lại cách sắp xếp không gian khi Conv2d chia patch.
        # Đây không phải phép đảo ngược trọng số học được của Conv2d.
        # Mỗi vị trí [56, 56] đang chứa 48 channel.
        # Tách 48 = 3 channel RGB x 4 pixel cao x 4 pixel rộng.
        batch_size, _, height, width = x.shape
        x = x.reshape(
            batch_size,
            3,
            self.patch_size,
            self.patch_size,
            height,
            width
        )
        # [B, 3, patch_h, patch_w, H, W]
        # -> [B, 3, H, patch_h, W, patch_w]
        x = x.permute(0, 1, 4, 2, 5, 3)

        # Ghép H với patch_h và W với patch_w.
        # [B, 3, 56, 4, 56, 4] -> [B, 3, 224, 224]
        x = x.reshape(
            batch_size,
            3,
            height * self.patch_size,
            width * self.patch_size
        )

        # Output của BiFormerEncoder: [B, 3, 224, 224]
        return x
