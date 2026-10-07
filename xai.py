"""
XAI CHO CHECKPOINT CÓ TEST ACCURACY CAO NHẤT
============================================

Phần 1 - PCA và t-SNE
1. Load checkpoint và test dataset.
2. Lấy feature sau GAP, trước MLP: [B, 256].
3. Chuẩn hóa feature bằng StandardScaler như cấu hình cũ.
4. PCA giảm trực tiếp xuống 2 chiều để vẽ PCA.
5. PCA giảm xuống 50 chiều rồi t-SNE giảm xuống 2 chiều.
6. Vẽ mỗi lớp bằng một màu khác nhau.

Phần 2 - Grad-CAM bằng thư viện pytorch-grad-cam
1. Target layer là GAMFusion cuối, ngay trước GAP.
2. GAM trả về token [B, 49, 256].
3. reshape_transform đổi token thành [B, 256, 7, 7].
4. Thư viện tính gradient của lớp dự đoán đối với feature GAM.
5. Heatmap được resize và chồng lên ảnh gốc.

Phần 3 - Hình so sánh giống bố cục bài báo
- BT-MRI nằm bên trái, Retinal nằm bên phải.
- Mỗi lớp chọn 4 ảnh test.
- Chỉ dùng checkpoint có test accuracy cao nhất của mỗi dataset.
- Hai cột cho mỗi dataset: Input và Grad-CAM.
"""

import csv
import inspect
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from main_evaluate import (
    CHECKPOINTS,
    CHECKPOINT_DIR,
    build_test_data,
    forward_with_features,
    load_model,
)
from utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "xai"
METRICS_CSV = (
    PROJECT_ROOT / "results" / "evaluation" / "metrics_all_checkpoints.csv"
)
NUMBER_OF_IMAGES_PER_CLASS = 4
RANDOM_SEED = 2026

# Thứ tự lớp để hình gần giống Figure 16 của bài báo.
CLASS_ORDER = {
    "bt_mri": ["glioma", "meningioma", "pituitary", "notumor"],
    "retinal": [
        "cataract",
        "diabetic_retinopathy",
        "glaucoma",
        "normal",
    ],
}

CLASS_DISPLAY_NAME = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor",
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "glaucoma": "Glaucoma",
    "normal": "Normal",
}


def extract_features(model, loader, device):
    """Lấy feature [N, 256] sau GAP và trước MLP."""
    labels_all = []
    features_all = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, _, gap_features = forward_with_features(model, images)

            labels_all.extend(labels.tolist())
            features_all.append(gap_features.cpu().numpy())

    labels_all = np.asarray(labels_all)
    features_all = np.concatenate(features_all)
    return labels_all, features_all


def draw_scatter(points, labels, classes, title, output_path):
    figure, axis = plt.subplots(figsize=(8, 7))
    colors = plt.get_cmap("tab10")

    for class_id, class_name in enumerate(classes):
        selected = labels == class_id
        axis.scatter(
            points[selected, 0],
            points[selected, 1],
            s=18,
            alpha=0.75,
            color=colors(class_id),
            label=CLASS_DISPLAY_NAME.get(class_name, class_name),
        )

    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.legend()
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_pca(features, labels, classes, output_path):
    """StandardScaler rồi PCA feature GAP xuống 2 chiều."""
    scaled_features = StandardScaler().fit_transform(features)
    points = PCA(
        n_components=2,
        random_state=42,
    ).fit_transform(scaled_features)
    draw_scatter(points, labels, classes, "PCA of GAP Features", output_path)


def create_tsne(features, labels, classes, output_path):
    """StandardScaler -> PCA 50 chiều -> t-SNE 2 chiều."""
    scaled_features = StandardScaler().fit_transform(features)

    number_of_components = min(
        50,
        scaled_features.shape[1],
        len(scaled_features) - 1,
    )
    pca_features = PCA(
        n_components=number_of_components,
        random_state=42,
    ).fit_transform(scaled_features)

    arguments = {
        "n_components": 2,
        "perplexity": 30,
        "metric": "cosine",
        "init": "pca",
        "learning_rate": "auto",
        "early_exaggeration": 12,
        "random_state": 42,
    }

    # scikit-learn mới dùng max_iter, bản cũ dùng n_iter.
    if "max_iter" in inspect.signature(TSNE).parameters:
        arguments["max_iter"] = 3000
    else:
        arguments["n_iter"] = 3000

    points = TSNE(**arguments).fit_transform(pca_features)
    draw_scatter(
        points,
        labels,
        classes,
        "t-SNE of GAP Features",
        output_path,
    )


def reshape_gam_tokens(tensor):
    """Đổi GAM token [B, 49, 256] về feature map [B, 256, 7, 7]."""
    batch_size, number_of_tokens, channels = tensor.shape
    side = int(math.sqrt(number_of_tokens))

    if side * side != number_of_tokens:
        raise ValueError("Số GAM token không tạo được feature map vuông")

    return tensor.transpose(1, 2).reshape(
        batch_size,
        channels,
        side,
        side,
    )


def create_gradcam_generator(model):
    """Tạo Grad-CAM và chọn GAMFusion làm target layer."""
    return GradCAM(
        model=model,
        target_layers=[model.fusion],
        reshape_transform=reshape_gam_tokens,
    )


def create_one_gradcam(model, gradcam, image, device):
    """Tạo heatmap cho một ảnh bằng lớp mà model dự đoán."""
    image_batch = image.unsqueeze(0).to(device)

    # Bước 1: lấy lớp dự đoán và confidence.
    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_class].item())

    # Bước 2: yêu cầu Grad-CAM giải thích lớp dự đoán.
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = gradcam(
        input_tensor=image_batch,
        targets=targets,
    )[0]

    # Bước 3: chồng heatmap lên ảnh RGB gốc.
    rgb_image = image.permute(1, 2, 0).numpy().astype(np.float32)
    rgb_image = np.clip(rgb_image, 0.0, 1.0)
    overlay = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.55,
    )

    return overlay, predicted_class, confidence


def select_four_images_per_class(dataset, class_order):
    """Chọn ngẫu nhiên 4 ảnh trong mỗi lớp với seed cố định."""
    indices_by_class = {}
    random_generator = random.Random(RANDOM_SEED)

    for class_name in class_order:
        if class_name not in dataset.class_to_idx:
            raise ValueError("Không tìm thấy class: " + class_name)

        class_id = dataset.class_to_idx[class_name]
        class_indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]
        if len(class_indices) < NUMBER_OF_IMAGES_PER_CLASS:
            raise ValueError(
                class_name + " không đủ 4 ảnh để tạo Grad-CAM"
            )

        indices_by_class[class_name] = random_generator.sample(
            class_indices,
            NUMBER_OF_IMAGES_PER_CLASS,
        )

    return indices_by_class


def create_all_gradcams(model, dataset, selected_indices, device):
    """Tạo Grad-CAM cho toàn bộ ảnh đã chọn của một checkpoint."""
    outputs = {}

    with create_gradcam_generator(model) as gradcam:
        for class_name, indices in selected_indices.items():
            outputs[class_name] = []

            for index in indices:
                image, _ = dataset[index]
                overlay, predicted_class, confidence = create_one_gradcam(
                    model,
                    gradcam,
                    image,
                    device,
                )
                outputs[class_name].append(
                    {
                        "overlay": overlay,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                    }
                )

    return outputs


def select_best_accuracy_configs():
    """Đọc bảng metric và chọn checkpoint có accuracy cao nhất mỗi dataset."""
    if not METRICS_CSV.exists():
        raise FileNotFoundError(
            "Chưa có bảng metric: " + str(METRICS_CSV) + "\n"
            "Hãy chạy: python main_evaluate.py"
        )

    best_rows = {}
    with METRICS_CSV.open("r", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            dataset_name = row["dataset"]
            accuracy = float(row["accuracy"])

            if dataset_name not in best_rows:
                best_rows[dataset_name] = row
            elif accuracy > float(best_rows[dataset_name]["accuracy"]):
                best_rows[dataset_name] = row

    selected_configs = {}
    for dataset_name in ["bt_mri", "retinal"]:
        if dataset_name not in best_rows:
            raise ValueError("Không có metric của dataset: " + dataset_name)

        checkpoint_kind = best_rows[dataset_name]["checkpoint"]
        matching_configs = [
            config
            for config in CHECKPOINTS
            if config["dataset"] == dataset_name
            and config["kind"] == checkpoint_kind
        ]
        if not matching_configs:
            raise ValueError(
                "Không tìm thấy cấu hình checkpoint: "
                + dataset_name + " / " + checkpoint_kind
            )

        selected_config = dict(matching_configs[0])
        selected_config["test_accuracy"] = float(
            best_rows[dataset_name]["accuracy"]
        )
        selected_configs[dataset_name] = selected_config

    return selected_configs


def prepare_dataset_xai(dataset_name, config, device):
    """Chỉ tạo XAI cho checkpoint có test accuracy cao nhất."""
    dataset, loader = build_test_data(config["test_folder"])
    classes = dataset.classes
    selected_indices = select_four_images_per_class(
        dataset,
        CLASS_ORDER[dataset_name],
    )

    checkpoint_path = CHECKPOINT_DIR / config["filename"]
    if not checkpoint_path.exists():
        raise FileNotFoundError("Không tìm thấy checkpoint: " + str(checkpoint_path))

    print("\nDataset:", dataset_name)
    print("Checkpoint được chọn:", checkpoint_path.name)
    print("Test accuracy:", f"{config['test_accuracy']:.4f}%")
    model, _ = load_model(checkpoint_path, classes, device)

    # PCA và t-SNE dùng feature của toàn bộ test set.
    print("  Bước 1: Trích xuất feature sau GAP")
    labels, features = extract_features(model, loader, device)

    output_folder = RESULTS_DIR / dataset_name / config["kind"]
    output_folder.mkdir(parents=True, exist_ok=True)

    print("  Bước 2: PCA")
    create_pca(
        features,
        labels,
        classes,
        output_folder / "pca.png",
    )

    print("  Bước 3: t-SNE")
    create_tsne(
        features,
        labels,
        classes,
        output_folder / "tsne.png",
    )

    print("  Bước 4: Grad-CAM bằng pytorch-grad-cam")
    gradcam_outputs = create_all_gradcams(
        model,
        dataset,
        selected_indices,
        device,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": dataset,
        "classes": classes,
        "config": config,
        "selected_indices": selected_indices,
        "gradcam_outputs": gradcam_outputs,
    }


def draw_paper_style_gradcam(all_results, output_path):
    """Vẽ BT-MRI và Retinal cạnh nhau giống bố cục Figure 16."""
    number_of_rows = 4 * NUMBER_OF_IMAGES_PER_CLASS
    figure, axes = plt.subplots(number_of_rows, 4, figsize=(9, 28))

    dataset_columns = {
        "bt_mri": 0,
        "retinal": 2,
    }

    for dataset_name, first_column in dataset_columns.items():
        result = all_results[dataset_name]
        dataset = result["dataset"]
        class_order = CLASS_ORDER[dataset_name]

        row = 0
        for class_name in class_order:
            indices = result["selected_indices"][class_name]

            for sample_number, index in enumerate(indices):
                image, _ = dataset[index]
                input_image = image.permute(1, 2, 0).numpy()

                axes[row, first_column].imshow(input_image)
                axes[row, first_column + 1].imshow(
                    result["gradcam_outputs"][class_name][
                        sample_number
                    ]["overlay"]
                )

                for column in range(first_column, first_column + 2):
                    axes[row, column].set_xticks([])
                    axes[row, column].set_yticks([])

                row += 1

            # Ghi tên lớp ở giữa nhóm 4 ảnh.
            middle_row = row - 2
            axes[middle_row, first_column].set_ylabel(
                CLASS_DISPLAY_NAME[class_name],
                fontsize=10,
                fontweight="bold",
                rotation=90,
                labelpad=16,
            )

            # Đường phân cách các lớp.
            if row < number_of_rows:
                for column in range(first_column, first_column + 2):
                    axes[row, column].spines["top"].set_visible(True)
                    axes[row, column].spines["top"].set_linewidth(1.5)

        axes[0, first_column].set_title("(a) Input", fontsize=11)
        axes[0, first_column + 1].set_title(
            "(b) Proposed Method",
            fontsize=11,
        )

    figure.text(
        0.27,
        0.995,
        "BT CE-MRI Dataset",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.76,
        0.995,
        "Retinal Dataset",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.975,
        bottom=0.01,
        wspace=0.04,
        hspace=0.04,
    )
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    selected_configs = select_best_accuracy_configs()

    all_results = {}
    for dataset_name in ["bt_mri", "retinal"]:
        all_results[dataset_name] = prepare_dataset_xai(
            dataset_name,
            selected_configs[dataset_name],
            device,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "gradcam_best_accuracy.png"
    draw_paper_style_gradcam(all_results, output_path)

    print("\nĐã lưu PCA, t-SNE và Grad-CAM tại:", RESULTS_DIR)
    print("Hình Grad-CAM tổng hợp:", output_path)


if __name__ == "__main__":
    main()
