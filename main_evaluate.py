import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data import build_eval_transform
from model import build_network
from model.modules.classifier import MLPGELUClassifier
from model.modules.cnn_backbones import (
    DenseNet121Backbone,
    EfficientNetB0Backbone,
)
from model.modules.fusion import GAMFusionNew
from utils import calculate_metrics, set_seed


# ================================================================
# 6 CHECKPOINT BEST ACCURACY CẦN ĐÁNH GIÁ
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation_cosine300"

BT_MRI_TEST_FOLDER = Path("/content/datasets/bt_mri/test")
RETINAL_TEST_FOLDER = Path("/content/datasets/retinal/test")

CHECKPOINTS = [
    {
        "dataset": "bt_mri",
        "kind": "Seed42",
        "test_folder": BT_MRI_TEST_FOLDER,
        "filename": (
            "checkpoint_qtrong_bt-mri/"
            "best_acc_bt_mri_GAMFusionNew_MLPGELUClassifier_"
            "FullDrop0p1_TwoStageLR_Cosine300_Seed42.pt"
        ),
    },
    {
        "dataset": "bt_mri",
        "kind": "Seed123",
        "test_folder": BT_MRI_TEST_FOLDER,
        "filename": (
            "checkpoint_qtrong_bt-mri/"
            "best_acc_bt_mri_GAMFusionNew_MLPGELUClassifier_"
            "FullDrop0p1_TwoStageLR_Cosine300_Seed123.pt"
        ),
    },
    {
        "dataset": "bt_mri",
        "kind": "Seed456",
        "test_folder": BT_MRI_TEST_FOLDER,
        "filename": (
            "checkpoint_qtrong_bt-mri/"
            "best_acc_bt_mri_GAMFusionNew_MLPGELUClassifier_"
            "FullDrop0p1_TwoStageLR_Cosine300_Seed456.pt"
        ),
    },
    {
        "dataset": "retinal",
        "kind": "Seed42",
        "test_folder": RETINAL_TEST_FOLDER,
        "filename": (
            "checkpoint_qtrong_rentinal/"
            "best_acc_retinal_GAMFusionNew_MLPGELUClassifier_"
            "FullDrop0p1_TwoStageLR_Cosine300_Seed42.pt"
        ),
    },
    {
        "dataset": "retinal",
        "kind": "Seed123",
        "test_folder": RETINAL_TEST_FOLDER,
        "filename": (
            "checkpoint_qtrong_rentinal/"
            "best_acc_retinal_GAMFusionNew_MLPGELUClassifier_"
            "FullDrop0p1_TwoStageLR_Cosine300_Seed123.pt"
        ),
    },
    {
        "dataset": "retinal",
        "kind": "Seed456",
        "test_folder": RETINAL_TEST_FOLDER,
        "filename": (
            "checkpoint_qtrong_rentinal/"
            "best_acc_retinal_GAMFusionNew_MLPGELUClassifier_"
            "FullDrop0p1_TwoStageLR_Cosine300_Seed456.pt"
        ),
    },
]

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

BIFORMER_DEPTH = 4
PATCH_SIZE = 4
N_WIN = 7
TOPK = 4
USE_PE = True

# Dropout không hoạt động khi model.eval(), nhưng giữ đúng cấu hình train.
DROPOUT = 0.1
MLP_DROPOUT = 0.1
DROP_PATH = 0.1

# ================================================================


MAIN_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "sensitivity",
    "specificity",
    "mcc",
    "auc",
]


def create_model(number_of_classes, device):
    model = build_network(
        num_classes=number_of_classes,
        biformer_depth=BIFORMER_DEPTH,
        patch_size=PATCH_SIZE,
        n_win=N_WIN,
        topk=TOPK,
        image_size=IMAGE_SIZE,
        use_pe=USE_PE,
        dropout=DROPOUT,
        mlp_dropout=MLP_DROPOUT,
        drop_path=DROP_PATH,
        cnn_1_class=DenseNet121Backbone,
        cnn_2_class=EfficientNetB0Backbone,
        fusion_class=GAMFusionNew,
        classifier_class=MLPGELUClassifier,
        fusion_dim=GAMFusionNew.default_embed_dim,
        fusion_heads=8,
        pretrained=False,
    )
    return model.to(device)


def load_model(checkpoint_path, classes, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_classes = checkpoint.get("classes")
    if checkpoint_classes is not None:
        if list(checkpoint_classes) != list(classes):
            raise ValueError(
                "Thứ tự class trong checkpoint khác test dataset.\n"
                + "Checkpoint: " + str(list(checkpoint_classes)) + "\n"
                + "Test data: " + str(list(classes))
            )

    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model = create_model(len(classes), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def build_test_data(test_folder):
    if not test_folder.exists():
        raise FileNotFoundError("Không tìm thấy test dataset: " + str(test_folder))

    transform = build_eval_transform(image_size=IMAGE_SIZE)
    dataset = ImageFolder(test_folder, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def forward_with_features(model, images):
    # Giữ lại feature GAM để file xai.py dùng cho PCA, t-SNE và Grad-CAM.
    images = model.encoder(images)
    feature_1 = model.cnn_1(images)
    feature_2 = model.cnn_2(images)
    fused_tokens = model.fusion(feature_1, feature_2)
    fused_vector = fused_tokens.mean(dim=1)
    logits = model.classifier(fused_vector)
    return logits, fused_tokens, fused_vector


def test_model(model, loader, device):
    labels_all = []
    predictions_all = []
    probabilities_all = []
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            total_loss += F.cross_entropy(
                logits,
                labels,
                reduction="sum",
            ).item()

            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            probabilities_all.append(probabilities.cpu().numpy())

    return {
        "test_loss": total_loss / len(loader.dataset),
        "labels": np.asarray(labels_all),
        "predictions": np.asarray(predictions_all),
        "probabilities": np.concatenate(probabilities_all),
    }


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def plot_confusion_matrix(matrix, classes, output_path):
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_xticks(np.arange(len(classes)), labels=classes, rotation=35)
    axis.set_yticks(np.arange(len(classes)), labels=classes)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion Matrix")

    threshold = matrix.max() / 2
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc_curve(labels, probabilities, classes, output_path):
    binary_labels = label_binarize(labels, classes=np.arange(len(classes)))
    figure, axis = plt.subplots(figsize=(7, 6))

    for class_id, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(
            binary_labels[:, class_id], probabilities[:, class_id]
        )
        axis.plot(fpr, tpr, label=class_name + f" ({auc(fpr, tpr):.3f})")

    fpr, tpr, _ = roc_curve(binary_labels.ravel(), probabilities.ravel())
    axis.plot(
        fpr,
        tpr,
        color="black",
        linestyle="--",
        label="Micro-average " + f"({auc(fpr, tpr):.3f})",
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


def plot_pr_curve(labels, probabilities, classes, output_path):
    binary_labels = label_binarize(labels, classes=np.arange(len(classes)))
    figure, axis = plt.subplots(figsize=(7, 6))

    for class_id, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(
            binary_labels[:, class_id], probabilities[:, class_id]
        )
        score = average_precision_score(
            binary_labels[:, class_id], probabilities[:, class_id]
        )
        axis.plot(recall, precision, label=class_name + f" ({score:.3f})")

    precision, recall, _ = precision_recall_curve(
        binary_labels.ravel(), probabilities.ravel()
    )
    score = average_precision_score(
        binary_labels, probabilities, average="micro"
    )
    axis.plot(
        recall,
        precision,
        color="black",
        linestyle="--",
        label="Micro-average " + f"({score:.3f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_percent_row(config, metrics):
    row = {
        "dataset": config["dataset"],
        "checkpoint": config["kind"],
    }
    for metric_name in MAIN_METRICS:
        value = metrics.get(metric_name)
        row[metric_name] = None if value is None else value * 100
    return row


def save_summary_table(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "metrics_all_checkpoints.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cell_text = []
    for row in rows:
        values = [row["dataset"], row["checkpoint"]]
        for metric_name in MAIN_METRICS:
            value = row[metric_name]
            values.append("N/A" if value is None else f"{value:.2f}")
        cell_text.append(values)

    columns = ["Dataset", "Checkpoint"] + [
        name.replace("_", " ").title() + " (%)" for name in MAIN_METRICS
    ]
    figure, axis = plt.subplots(figsize=(17, 3.6))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)
    axis.set_title("Test Metrics", fontsize=14)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "metrics_table.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    summary_rows = []

    for config in CHECKPOINTS:
        checkpoint_path = CHECKPOINT_DIR / config["filename"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Không tìm thấy checkpoint: " + str(checkpoint_path)
            )

        dataset, loader = build_test_data(config["test_folder"])
        classes = dataset.classes
        model, checkpoint = load_model(checkpoint_path, classes, device)

        print("\nDataset:", config["dataset"], "|", config["kind"])
        print("Checkpoint:", checkpoint_path.name)
        print("Classes:", classes)
        print("Đang đánh giá trên", len(dataset), "ảnh test...")

        evaluation = test_model(model, loader, device)
        metrics = calculate_metrics(
            evaluation["labels"],
            evaluation["predictions"],
            probabilities=evaluation["probabilities"],
            class_names=classes,
        )
        metrics_percent = {
            name: None if metrics.get(name) is None else metrics[name] * 100
            for name in MAIN_METRICS
        }

        output_folder = (
            RESULTS_DIR / config["dataset"] / config["kind"]
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset": config["dataset"],
            "seed": config["kind"],
            "checkpoint": checkpoint_path.name,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "validation_loss": checkpoint.get(
                "val_loss",
                checkpoint.get("best_val_loss"),
            ),
            "validation_accuracy": checkpoint.get(
                "val_accuracy",
                checkpoint.get("best_val_accuracy"),
            ),
            "test_loss": evaluation["test_loss"],
            "classes": classes,
            "metrics": metrics,
            "metrics_percent": metrics_percent,
        }
        save_json(output_folder / "metrics.json", report)

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            classes,
            output_folder / "confusion_matrix.png",
        )
        plot_roc_curve(
            evaluation["labels"],
            evaluation["probabilities"],
            classes,
            output_folder / "roc_curve.png",
        )
        plot_pr_curve(
            evaluation["labels"],
            evaluation["probabilities"],
            classes,
            output_folder / "precision_recall_curve.png",
        )

        row = make_percent_row(config, metrics)
        summary_rows.append(row)

        print("KẾT QUẢ TEST (%)")
        print("Accuracy:   ", f"{metrics_percent['accuracy']:.4f}")
        print("Precision:  ", f"{metrics_percent['precision']:.4f}")
        print("Recall:     ", f"{metrics_percent['recall']:.4f}")
        print("F1-Score:   ", f"{metrics_percent['f1_score']:.4f}")
        print("Sensitivity:", f"{metrics_percent['sensitivity']:.4f}")
        print("Specificity:", f"{metrics_percent['specificity']:.4f}")
        print("MCC:        ", f"{metrics_percent['mcc']:.4f}")
        auc_value = metrics_percent["auc"]
        print("AUC:        ", "N/A" if auc_value is None else f"{auc_value:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_summary_table(summary_rows)

    print("\nĐã lưu kết quả tại:", RESULTS_DIR)


if __name__ == "__main__":
    main()
