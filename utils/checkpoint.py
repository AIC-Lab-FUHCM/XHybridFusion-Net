import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(model, path, epoch, val_loss, val_accuracy, classes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "classes": classes,
        },
        path,
    )


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_training_checkpoint(
    model,
    optimizer,
    path,
    epoch,
    best_val_loss,
    best_val_accuracy,
    early_stop_best_loss,
    classes,
    early_stop_counter,
    scheduler=None,
    data_generator=None,
):
    """Save the complete state required to resume interrupted training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "early_stop_best_loss": early_stop_best_loss,
        "early_stop_counter": early_stop_counter,
        "classes": classes,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_random_state"] = torch.cuda.get_rng_state_all()

    if data_generator is not None:
        checkpoint["data_generator_state"] = data_generator.get_state()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_training_checkpoint(
    model,
    optimizer,
    path,
    device,
    scheduler=None,
    data_generator=None,
):
    """Restore model, optimizer, scheduler, and random-number states."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore the shuffle, augmentation, and dropout random sequences.
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
        cuda_states = [state.cpu() for state in checkpoint["cuda_random_state"]]
        torch.cuda.set_rng_state_all(cuda_states)
    if data_generator is not None and "data_generator_state" in checkpoint:
        data_generator.set_state(checkpoint["data_generator_state"].cpu())

    return checkpoint
