import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import build_dataset


def seed_worker(worker_id):
    """Seed NumPy and Python inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    data_root,
    train_transform,
    eval_transform,
    batch_size=8,
    num_workers=0,
    seed=42,
):
    """Build reproducible training, validation, and test data loaders."""
    data_root = Path(data_root)

    train_dataset = build_dataset(data_root / "train", train_transform)
    val_dataset = build_dataset(data_root / "val", eval_transform)
    test_dataset = build_dataset(data_root / "test", eval_transform)

    # This generator controls the training-set shuffle order.
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=train_generator,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "classes": train_dataset.classes,
        "train_generator": train_generator,
    }
