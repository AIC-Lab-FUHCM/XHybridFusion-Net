import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data import build_eval_transform
from utils import calculate_metrics, set_seed
from utils.config import (
    build_model_from_config,
    load_config,
    resolve_path,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "sensitivity",
    "specificity",
    "mcc",
    "auc",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one trained checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Override data_root from the config")
    parser.add_argument("--output-dir", help="Directory for metrics and figures")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_model(config, checkpoint_path, classes, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_classes = checkpoint.get("classes")
    if checkpoint_classes is not None and list(checkpoint_classes) != list(classes):
        raise ValueError(
            "Class order does not match the checkpoint. Checkpoint="
            + str(list(checkpoint_classes))
            + ", dataset="
            + str(list(classes))
        )

    model = build_model_from_config(
        config,
        number_of_classes=len(classes),
        device=device,
        pretrained=False,
    )
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model, loader, device):
    labels_all = []
    predictions_all = []
    probabilities_all = []
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

        total_loss += F.cross_entropy(
            logits, labels, reduction="sum"
        ).item()
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
        probabilities_all.append(probabilities.cpu().numpy())

    return {
        "loss": total_loss / len(loader.dataset),
        "labels": np.asarray(labels_all),
        "predictions": np.asarray(predictions_all),
        "probabilities": np.concatenate(probabilities_all),
    }


def plot_confusion_matrix(matrix, classes, output_path):
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc(labels, probabilities, classes, output_path):
    binary_labels = label_binarize(labels, classes=np.arange(len(classes)))
    figure, axis = plt.subplots(figsize=(7, 6))

    for class_id, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(
            binary_labels[:, class_id], probabilities[:, class_id]
        )
        axis.plot(fpr, tpr, label=f"{class_name} ({auc(fpr, tpr):.6f})")

    fpr, tpr, _ = roc_curve(binary_labels.ravel(), probabilities.ravel())
    axis.plot(
        fpr,
        tpr,
        color="black",
        linestyle="--",
        label=f"Micro-average ({auc(fpr, tpr):.6f})",
    )
    axis.plot([0, 1], [0, 1], color="gray", linestyle=":")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_precision_recall(labels, probabilities, classes, output_path):
    binary_labels = label_binarize(labels, classes=np.arange(len(classes)))
    figure, axis = plt.subplots(figsize=(7, 6))

    for class_id, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(
            binary_labels[:, class_id], probabilities[:, class_id]
        )
        axis.plot(recall, precision, label=class_name)

    precision, recall, _ = precision_recall_curve(
        binary_labels.ravel(), probabilities.ravel()
    )
    axis.plot(recall, precision, color="black", linestyle="--", label="Micro-average")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    if args.data_root:
        config["data_root"] = args.data_root

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_path(config["data_root"], PROJECT_ROOT)
    test_folder = data_root / "test"
    checkpoint_path = resolve_path(args.checkpoint, PROJECT_ROOT)
    output_dir = resolve_path(
        args.output_dir
        or f"outputs/evaluation/{config['dataset']}/{checkpoint_path.stem}",
        PROJECT_ROOT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolder(
        test_folder,
        transform=build_eval_transform(config["model"]["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model, checkpoint = load_model(
        config, checkpoint_path, dataset.classes, device
    )

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Test images:", len(dataset))
    evaluation = predict(model, loader, device)
    metrics = calculate_metrics(
        evaluation["labels"],
        evaluation["predictions"],
        probabilities=evaluation["probabilities"],
        class_names=dataset.classes,
    )

    report = {
        "dataset": config["dataset"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_loss": checkpoint.get("val_loss", checkpoint.get("best_val_loss")),
        "validation_accuracy": checkpoint.get(
            "val_accuracy", checkpoint.get("best_val_accuracy")
        ),
        "test_loss": evaluation["loss"],
        "classes": dataset.classes,
        "metrics": metrics,
        "metrics_percent": {
            name: None if metrics.get(name) is None else metrics[name] * 100
            for name in MAIN_METRICS
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["dataset", "checkpoint", *MAIN_METRICS])
        writer.writeheader()
        writer.writerow(
            {
                "dataset": config["dataset"],
                "checkpoint": checkpoint_path.name,
                **report["metrics_percent"],
            }
        )

    plot_confusion_matrix(
        metrics["confusion_matrix"], dataset.classes, output_dir / "confusion_matrix.png"
    )
    plot_roc(
        evaluation["labels"],
        evaluation["probabilities"],
        dataset.classes,
        output_dir / "roc_curve.png",
    )
    plot_precision_recall(
        evaluation["labels"],
        evaluation["probabilities"],
        dataset.classes,
        output_dir / "precision_recall_curve.png",
    )

    print("\nTest results (%)")
    for name in MAIN_METRICS:
        value = report["metrics_percent"][name]
        print(f"{name:12s}: {'N/A' if value is None else f'{value:.6f}'}")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
