import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from data import build_dataloaders, build_eval_transform, build_train_transform
from utils import (
    calculate_metrics,
    evaluate,
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    set_seed,
    train_one_epoch,
)
from utils.config import (
    build_model_from_config,
    load_config,
    resolve_path,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train BiFormer-GAM medical classifier")
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    parser.add_argument("--data-root", help="Override data_root from the config")
    parser.add_argument("--output-root", help="Override output_root from the config")
    parser.add_argument("--no-resume", action="store_true", help="Do not load last.pt")
    return parser.parse_args()


def set_learning_rates(optimizer, epoch, training):
    if epoch <= training["first_stage_epochs"]:
        learning_rates = {
            "biformer": training["first_stage_biformer_lr"],
            "cnn_1": training["first_stage_cnn_lr"],
            "cnn_2": training["first_stage_cnn_lr"],
            "fusion": training["first_stage_fusion_lr"],
            "classifier": training["first_stage_classifier_lr"],
        }
    else:
        learning_rates = {
            name: training["second_stage_lr"]
            for name in ("biformer", "cnn_1", "cnn_2", "fusion", "classifier")
        }

    for group in optimizer.param_groups:
        group["lr"] = learning_rates[group["name"]]


def learning_rate_text(optimizer):
    return ", ".join(
        group["name"] + "=" + f"{group['lr']:.1e}"
        for group in optimizer.param_groups
    )


def save_test_result(model, checkpoint_path, data, criterion, device, output_path):
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    test_loss, labels, predictions = evaluate(
        model, data["test"], criterion, device
    )
    metrics = calculate_metrics(labels, predictions)
    metrics.update(
        {
            "loss": test_loss,
            "checkpoint_epoch": checkpoint["epoch"],
            "validation_loss": checkpoint["val_loss"],
            "validation_accuracy": checkpoint["val_accuracy"],
        }
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    if args.data_root:
        config["data_root"] = args.data_root
    if args.output_root:
        config["output_root"] = args.output_root

    model_config = config["model"]
    training = config["training"]
    seed = config["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_path(config["data_root"], PROJECT_ROOT)
    output_root = resolve_path(config.get("output_root", "outputs"), PROJECT_ROOT)
    run_dir = output_root / config["dataset"] / config["run_name"]
    checkpoint_dir = run_dir / "checkpoints"
    result_dir = run_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Dataset:", config["dataset"])
    print("Data root:", data_root)
    print("Run:", config["run_name"])

    train_transform = build_train_transform(model_config["image_size"])
    eval_transform = build_eval_transform(model_config["image_size"])
    data = build_dataloaders(
        data_root=data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        seed=seed,
    )

    model = build_model_from_config(
        config,
        number_of_classes=len(data["classes"]),
        device=device,
    )
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": training["first_stage_biformer_lr"],
                "name": "biformer",
            },
            {
                "params": model.cnn_1.parameters(),
                "lr": training["first_stage_cnn_lr"],
                "name": "cnn_1",
            },
            {
                "params": model.cnn_2.parameters(),
                "lr": training["first_stage_cnn_lr"],
                "name": "cnn_2",
            },
            {
                "params": model.fusion.parameters(),
                "lr": training["first_stage_fusion_lr"],
                "name": "fusion",
            },
            {
                "params": model.classifier.parameters(),
                "lr": training["first_stage_classifier_lr"],
                "name": "classifier",
            },
        ],
        weight_decay=training["weight_decay"],
    )

    for group in optimizer.param_groups:
        group["lr"] = training["second_stage_lr"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training["epochs"] - training["first_stage_epochs"],
        eta_min=training["min_lr"],
    )

    best_loss_path = checkpoint_dir / "best_loss.pt"
    best_accuracy_path = checkpoint_dir / "best_accuracy.pt"
    last_path = checkpoint_dir / "last.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_accuracy = float("-inf")
    early_stop_best_loss = float("inf")
    early_stop_counter = 0
    resume = training.get("resume", True) and not args.no_resume

    if resume and last_path.exists():
        checkpoint = load_training_checkpoint(
            model,
            optimizer,
            last_path,
            device,
            scheduler=scheduler,
            data_generator=data["train_generator"],
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_val_accuracy = checkpoint.get("best_val_accuracy", float("-inf"))
        early_stop_best_loss = checkpoint.get("early_stop_best_loss", best_val_loss)
        early_stop_counter = checkpoint.get("early_stop_counter", 0)
        print("Resuming from epoch", start_epoch)

    for epoch in range(start_epoch, training["epochs"] + 1):
        if epoch <= training["first_stage_epochs"] + 1:
            set_learning_rates(optimizer, epoch, training)

        train_loss, train_accuracy = train_one_epoch(
            model, data["train"], criterion, optimizer, device
        )
        val_loss, val_labels, val_predictions = evaluate(
            model, data["val"], criterion, device
        )
        val_accuracy = calculate_metrics(val_labels, val_predictions)["accuracy"]

        print(
            f"Epoch {epoch:03d}/{training['epochs']} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f} | "
            f"lr: {learning_rate_text(optimizer)}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, best_loss_path, epoch, val_loss, val_accuracy, data["classes"]
            )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model,
                best_accuracy_path,
                epoch,
                val_loss,
                val_accuracy,
                data["classes"],
            )

        if training.get("early_stopping", False):
            if val_loss < early_stop_best_loss - training["min_delta"]:
                early_stop_best_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(
                    "Early stopping:",
                    f"{early_stop_counter}/{training['patience']}",
                )
        else:
            early_stop_best_loss = min(early_stop_best_loss, val_loss)
            early_stop_counter = 0

        if epoch > training["first_stage_epochs"]:
            scheduler.step()

        should_stop = (
            training.get("early_stopping", False)
            and early_stop_counter >= training["patience"]
        )
        if (
            epoch % training["save_last_every"] == 0
            or epoch == training["epochs"]
            or should_stop
        ):
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                path=last_path,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_val_accuracy=best_val_accuracy,
                early_stop_best_loss=early_stop_best_loss,
                classes=data["classes"],
                early_stop_counter=early_stop_counter,
                scheduler=scheduler,
                data_generator=data["train_generator"],
            )

        if should_stop:
            print("Early stopping at epoch", epoch)
            break

    best_loss_metrics = save_test_result(
        model,
        best_loss_path,
        data,
        criterion,
        device,
        result_dir / "metrics_best_loss.json",
    )
    best_accuracy_metrics = save_test_result(
        model,
        best_accuracy_path,
        data,
        criterion,
        device,
        result_dir / "metrics_best_accuracy.json",
    )
    print("\nBest-loss test accuracy:", best_loss_metrics["accuracy"])
    print("Best-accuracy test accuracy:", best_accuracy_metrics["accuracy"])
    print("Results:", run_dir)


if __name__ == "__main__":
    main()
