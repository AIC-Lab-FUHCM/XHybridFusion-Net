import os
import random

import numpy as np

# Phải đặt trước khi CUDA được khởi tạo.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Không cho cuDNN tự chọn thuật toán khác nhau giữa các lần chạy.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Dừng ngay nếu model gọi phép toán CUDA không deterministic.
    torch.use_deterministic_algorithms(True, warn_only=False)

    # Giữ cách tính float32 giống nhau giữa các lần chạy.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
