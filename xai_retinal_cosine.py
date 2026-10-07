"""XAI cho các checkpoint retinal Cosine 300 epoch.

Grad-CAM:
- Target là model.fusion.output_projection.
- Feature tại đây có shape [B, 768, 7, 7].
- Đây là output sau Conv 1x1 + BatchNorm + ReLU và trước flatten/GAP.

PCA và t-SNE:
- Lấy vector sau GAP, trước classifier, shape [B, 768].
"""

import inspect
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
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data import build_eval_transform
from model import build_network
from model.modules.classifier import MLPGELUClassifier
from model.modules.cnn_backbones import DenseNet121Backbone, EfficientNetB0Backbone
from model.modules.fusion import GAMFusionNew


# ================================================================
# CẤU HÌNH
# ================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
TEST_FOLDER = Path("/content/datasets/retinal/test")
OUTPUT_DIR = PROJECT_ROOT / "results" / "xai_retinal_cosine"

CHECKPOINT_FILENAMES = [
    (
        "best_acc_retinal_GAMFusionNew_MLPGELUClassifier_"
        "FullDrop0p1_TwoStageLR_Cosine300_Seed456.pt"
    ),
    (
        "best_acc_retinal_GAMFusionNew_MLPGELUClassifier_"
        "FullDrop0p1_TwoStageLR_Cosine300_Seed123.pt"
    ),
    (
        "best_acc_retinal_GAMFusionNew_MLPGELUClassifier_"
        "FullDrop0p1_TwoStageLR_Cosine300_Seed42.pt"
    ),
]

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 2026
IMAGES_PER_CLASS = 4

BIFORMER_DEPTH = 4
PATCH_SIZE = 4
N_WIN = 7
TOPK = 4
USE_PE = True
DROPOUT = 0.1
MLP_DROPOUT = 0.1
DROP_PATH = 0.1
# ================================================================


def set_xai_seed(seed):
    """Cố định chọn ảnh và phép giảm chiều, không khóa CUDA backward."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_checkpoint():
    """Chọn checkpoint có validation accuracy cao nhất trong 3 seed."""
    candidates = []

    for filename in CHECKPOINT_FILENAMES:
        path = CHECKPOINT_DIR / filename
        if not path.exists():
            print("Bỏ qua vì chưa có checkpoint:", path.name)
            continue

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validation_accuracy = checkpoint.get("val_accuracy", float("-inf"))
        candidates.append((float(validation_accuracy), path))

    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy checkpoint retinal cosine trong "
            + str(CHECKPOINT_DIR)
        )

    validation_accuracy, checkpoint_path = max(candidates, key=lambda item: item[0])
    print("Checkpoint XAI:", checkpoint_path.name)
    print("Validation accuracy:", validation_accuracy)
    return checkpoint_path


def build_test_data():
    if not TEST_FOLDER.exists():
        raise FileNotFoundError("Không tìm thấy test dataset: " + str(TEST_FOLDER))

    dataset = ImageFolder(
        TEST_FOLDER,
        transform=build_eval_transform(image_size=IMAGE_SIZE),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def create_model(number_of_classes, device):
    """Dựng đúng kiến trúc đã dùng để train checkpoint mới."""
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
        cnn_1_freeze_blocks=0,
        cnn_2_freeze_blocks=0,
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
    return model


def forward_with_gap(model, images):
    """Forward thủ công để lấy vector sau GAP và trước classifier."""
    images = model.encoder(images)
    feature_1 = model.cnn_1(images)
    feature_2 = model.cnn_2(images)
    fused_tokens = model.fusion(feature_1, feature_2)
    gap_features = fused_tokens.mean(dim=1)
    logits = model.classifier(gap_features)
    return logits, gap_features


@torch.no_grad()
def extract_gap_features(model, loader, device):
    """Lấy feature GAP [N, 768] để vẽ PCA và t-SNE."""
    labels_all = []
    features_all = []

    model.eval()
    for images, labels in loader:
        images = images.to(device)
        _, gap_features = forward_with_gap(model, images)
        labels_all.extend(labels.tolist())
        features_all.append(gap_features.cpu().numpy())

    return np.asarray(labels_all), np.concatenate(features_all)


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
            label=class_name.replace("_", " ").title(),
        )

    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.legend()
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_pca_and_tsne(features, labels, classes):
    """StandardScaler -> PCA/t-SNE từ feature GAP."""
    scaled_features = StandardScaler().fit_transform(features)

    pca_points = PCA(
        n_components=2,
        random_state=RANDOM_SEED,
    ).fit_transform(scaled_features)
    draw_scatter(
        pca_points,
        labels,
        classes,
        "PCA of GAP Features",
        OUTPUT_DIR / "pca_gap.png",
    )

    pca_dimensions = min(50, scaled_features.shape[0] - 1, scaled_features.shape[1])
    reduced_features = PCA(
        n_components=pca_dimensions,
        random_state=RANDOM_SEED,
    ).fit_transform(scaled_features)

    perplexity = min(30.0, max(5.0, (len(labels) - 1) / 3.0))
    arguments = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": RANDOM_SEED,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        arguments["max_iter"] = 1000
    else:
        arguments["n_iter"] = 1000

    tsne_points = TSNE(**arguments).fit_transform(reduced_features)
    draw_scatter(
        tsne_points,
        labels,
        classes,
        "t-SNE of GAP Features",
        OUTPUT_DIR / "tsne_gap.png",
    )


def select_images_per_class(dataset):
    """Chọn ngẫu nhiên 4 ảnh mỗi lớp với seed cố định."""
    generator = random.Random(RANDOM_SEED)
    selected = {}

    for class_id, class_name in enumerate(dataset.classes):
        indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]
        count = min(IMAGES_PER_CLASS, len(indices))
        selected[class_name] = generator.sample(indices, count)

    return selected


def create_gradcam_figure(model, dataset, selected_indices, device):
    """Grad-CAM tại output_projection, trước flatten và GAP."""
    rows = len(dataset.classes)
    columns = IMAGES_PER_CLASS * 2
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.0, rows * 2.3),
        squeeze=False,
    )

    # output_projection trả [B, 768, 7, 7], nên không cần reshape_transform.
    with GradCAM(
        model=model,
        target_layers=[model.fusion.output_projection],
    ) as gradcam:
        for row, class_name in enumerate(dataset.classes):
            indices = selected_indices[class_name]

            for image_number, index in enumerate(indices):
                image, _ = dataset[index]
                image_batch = image.unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(image_batch)
                    predicted_class = int(logits.argmax(dim=1).item())

                heatmap = gradcam(
                    input_tensor=image_batch,
                    targets=[ClassifierOutputTarget(predicted_class)],
                )[0]

                rgb_image = image.permute(1, 2, 0).numpy().astype(np.float32)
                rgb_image = np.clip(rgb_image, 0.0, 1.0)
                overlay = show_cam_on_image(
                    rgb_image,
                    heatmap,
                    use_rgb=True,
                    image_weight=0.55,
                )

                input_axis = axes[row, image_number * 2]
                cam_axis = axes[row, image_number * 2 + 1]
                input_axis.imshow(rgb_image)
                cam_axis.imshow(overlay)

                if row == 0:
                    input_axis.set_title("Input")
                    cam_axis.set_title("Proposed Method")

                input_axis.axis("off")
                cam_axis.axis("off")

            axes[row, 0].set_ylabel(
                class_name.replace("_", " ").title(),
                rotation=90,
                fontsize=10,
            )

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "gradcam_output_projection.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    set_xai_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint_path = choose_checkpoint()
    dataset, loader = build_test_data()
    model = load_model(checkpoint_path, dataset.classes, device)

    print("Bước 1: lấy feature sau GAP")
    labels, gap_features = extract_gap_features(model, loader, device)
    print("GAP feature shape:", gap_features.shape)

    print("Bước 2: vẽ PCA và t-SNE")
    create_pca_and_tsne(gap_features, labels, dataset.classes)

    print("Bước 3: vẽ Grad-CAM tại fusion.output_projection")
    selected_indices = select_images_per_class(dataset)
    create_gradcam_figure(model, dataset, selected_indices, device)

    print("Đã lưu XAI tại:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
