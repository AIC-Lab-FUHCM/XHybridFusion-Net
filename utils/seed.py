import os
import random

import numpy as np

# This variable must be set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Prevent cuDNN from selecting different algorithms between runs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Fail if the model invokes a nondeterministic CUDA operation.
    torch.use_deterministic_algorithms(True, warn_only=False)

    # Keep float32 computation settings consistent between runs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
