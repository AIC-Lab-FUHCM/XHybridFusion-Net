"""XAI cho checkpoint best accuracy BT-MRI, seed 42.

Grad-CAM:
- Vẽ riêng cho cả BT-MRI và retinal bằng checkpoint tương ứng.
- Target là model.fusion.output_projection.
- Feature tại đây có shape [B, 768, 7, 7].
- Đây là output sau Conv 1x1 + BatchNorm + ReLU và trước flatten/GAP.

t-SNE:
- BT-MRI dùng checkpoint best accuracy seed 42.
- Retinal dùng checkpoint best accuracy seed 42.
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
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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
BT_MRI_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoint_qtrong_bt-mri"
    / (
        "best_acc_bt_mri_GAMFusionNew_MLPGELUClassifier_"
        "FullDrop0p1_TwoStageLR_Cosine300_Seed42.pt"
    )
)
BT_MRI_TEST_FOLDER = Path("/content/datasets/bt_mri/test")
RETINAL_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoint_qtrong_rentinal"
    / (
        "best_acc_retinal_GAMFusionNew_MLPGELUClassifier_"
        "FullDrop0p1_TwoStageLR_Cosine300_Seed42.pt"
    )
)
RETINAL_TEST_FOLDER = Path("/content/datasets/retinal/test")
OUTPUT_DIR = PROJECT_ROOT / "results" / "xai_bt_mri_gradcam_retinal_tsne"

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


def choose_bt_mri_checkpoint():
    """Chỉ nhận đúng checkpoint best accuracy của BT-MRI seed 42."""
    if not BT_MRI_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Không tìm thấy checkpoint BT-MRI: "
            + str(BT_MRI_CHECKPOINT_PATH)
        )

    checkpoint_path = BT_MRI_CHECKPOINT_PATH
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    print("Checkpoint XAI:", checkpoint_path.name)
    print("Validation accuracy:", checkpoint.get("val_accuracy"))
    return checkpoint_path


def build_test_data(test_folder):
    if not test_folder.exists():
        raise FileNotFoundError("Không tìm thấy test dataset: " + str(test_folder))

    dataset = ImageFolder(
        test_folder,
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
    """Lấy feature GAP [N, 768] để vẽ t-SNE."""
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


def create_tsne(
    features,
    labels,
    classes,
    dataset_display_name,
    output_filename,
):
    """StandardScaler -> t-SNE trực tiếp từ feature GAP, không dùng PCA."""
    scaled_features = StandardScaler().fit_transform(features)

    perplexity = min(30.0, max(5.0, (len(labels) - 1) / 3.0))
    arguments = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "random",
        "learning_rate": "auto",
        "random_state": RANDOM_SEED,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        arguments["max_iter"] = 1000
    else:
        arguments["n_iter"] = 1000

    tsne_points = TSNE(**arguments).fit_transform(scaled_features)
    draw_scatter(
        tsne_points,
        labels,
        classes,
        "t-SNE of " + dataset_display_name + " GAP Features",
        OUTPUT_DIR / output_filename,
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


def create_cam_outputs(model, dataset, selected_indices, device, cam_class):
    """Tạo heatmap của một phương pháp CAM cho các ảnh đã chọn."""
    outputs = {}

    # output_projection trả [B, 768, 7, 7], nên không cần reshape_transform.
    with cam_class(
        model=model,
        target_layers=[model.fusion.output_projection],
    ) as cam_generator:
        for class_name in dataset.classes:
            outputs[class_name] = []

            for index in selected_indices[class_name]:
                image, _ = dataset[index]
                image_batch = image.unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(image_batch)
                    predicted_class = int(logits.argmax(dim=1).item())

                heatmap = cam_generator(
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
                outputs[class_name].append(
                    {
                        "input": rgb_image,
                        "cam": overlay,
                    }
                )

    return outputs


def class_display_name(class_name):
    names = {
        "glioma": "Glioma Tumor",
        "meningioma": "Meningioma Tumor",
        "pituitary": "Pituitary Tumor",
        "notumor": "No Tumor",
        "no_tumor": "No Tumor",
        "cataract": "Cataract",
        "diabetic_retinopathy": "Diabetic Retinopathy",
        "glaucoma": "Glaucoma",
        "normal": "Normal",
    }
    return names.get(class_name, class_name.replace("_", " ").title())


def draw_vertical_cam_comparison(
    bt_mri_dataset,
    bt_mri_gradcam,
    bt_mri_gradcam_plus_plus,
    retinal_dataset,
    retinal_gradcam,
    retinal_gradcam_plus_plus,
):
    """Hai dataset nằm cạnh nhau, mỗi dataset có ba cột CAM."""
    bt_rows = sum(len(bt_mri_gradcam[name]) for name in bt_mri_dataset.classes)
    retinal_rows = sum(
        len(retinal_gradcam[name]) for name in retinal_dataset.classes
    )
    total_rows = max(bt_rows, retinal_rows)

    figure, axes = plt.subplots(
        total_rows,
        6,
        figsize=(12, total_rows * 1.45),
        squeeze=False,
    )

    for axis_row in axes:
        for axis in axis_row:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    panels = [
        (
            "BT CE-MRI Dataset",
            bt_mri_dataset.classes,
            bt_mri_gradcam,
            bt_mri_gradcam_plus_plus,
            0,
        ),
        (
            "Retinal Dataset",
            retinal_dataset.classes,
            retinal_gradcam,
            retinal_gradcam_plus_plus,
            3,
        ),
    ]

    column_titles = ["(a) Input", "(b) Grad-CAM", "(c) Grad-CAM++"]

    for dataset_title, classes, gradcam, gradcam_plus_plus, first_column in panels:
        current_row = 0

        for class_name in classes:
            class_start_row = current_row
            gradcam_items = gradcam[class_name]
            plus_plus_items = gradcam_plus_plus[class_name]

            for item_number, gradcam_item in enumerate(gradcam_items):
                plus_plus_item = plus_plus_items[item_number]
                axes[current_row, first_column].imshow(gradcam_item["input"])
                axes[current_row, first_column + 1].imshow(gradcam_item["cam"])
                axes[current_row, first_column + 2].imshow(plus_plus_item["cam"])
                current_row += 1

            middle_row = class_start_row + (len(gradcam_items) - 1) // 2
            axes[middle_row, first_column].set_ylabel(
                class_display_name(class_name),
                rotation=90,
                fontsize=9,
                labelpad=13,
            )

            # Đường ngang chia các nhóm bệnh giống bố cục bài báo.
            if class_start_row > 0:
                for column in range(first_column, first_column + 3):
                    axes[class_start_row, column].plot(
                        [0, 1],
                        [1.02, 1.02],
                        transform=axes[class_start_row, column].transAxes,
                        color="black",
                        linewidth=1.0,
                        clip_on=False,
                    )

        for offset, title in enumerate(column_titles):
            axes[0, first_column + offset].set_title(title, fontsize=10, pad=7)

        center_column = first_column + 1
        axes[0, center_column].text(
            0.5,
            1.35,
            dataset_title,
            transform=axes[0, center_column].transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.965,
        bottom=0.01,
        wspace=0.04,
        hspace=0.04,
    )
    figure.savefig(
        OUTPUT_DIR / "gradcam_gradcamplusplus_vertical.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    set_xai_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Bước 1: load BT-MRI best accuracy seed 42")
    bt_mri_checkpoint_path = choose_bt_mri_checkpoint()
    bt_mri_dataset, bt_mri_loader = build_test_data(BT_MRI_TEST_FOLDER)
    bt_mri_model = load_model(
        bt_mri_checkpoint_path,
        bt_mri_dataset.classes,
        device,
    )

    print("Bước 2: lấy feature GAP và vẽ t-SNE BT-MRI")
    bt_mri_labels, bt_mri_gap_features = extract_gap_features(
        bt_mri_model,
        bt_mri_loader,
        device,
    )
    print("BT-MRI GAP feature shape:", bt_mri_gap_features.shape)
    create_tsne(
        bt_mri_gap_features,
        bt_mri_labels,
        bt_mri_dataset.classes,
        dataset_display_name="BT-MRI",
        output_filename="tsne_bt_mri_gap.png",
    )

    print("Bước 3: tạo Grad-CAM BT-MRI")
    bt_mri_selected_indices = select_images_per_class(bt_mri_dataset)
    bt_mri_gradcam = create_cam_outputs(
        bt_mri_model,
        bt_mri_dataset,
        bt_mri_selected_indices,
        device,
        cam_class=GradCAM,
    )

    print("Bước 4: tạo Grad-CAM++ BT-MRI")
    bt_mri_gradcam_plus_plus = create_cam_outputs(
        bt_mri_model,
        bt_mri_dataset,
        bt_mri_selected_indices,
        device,
        cam_class=GradCAMPlusPlus,
    )

    del bt_mri_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Bước 5: load retinal best accuracy seed 42")
    if not RETINAL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Không tìm thấy checkpoint retinal: "
            + str(RETINAL_CHECKPOINT_PATH)
        )
    retinal_dataset, retinal_loader = build_test_data(RETINAL_TEST_FOLDER)
    retinal_model = load_model(
        RETINAL_CHECKPOINT_PATH,
        retinal_dataset.classes,
        device,
    )

    print("Bước 6: lấy feature GAP và vẽ t-SNE retinal")
    labels, gap_features = extract_gap_features(
        retinal_model,
        retinal_loader,
        device,
    )
    print("Retinal GAP feature shape:", gap_features.shape)
    create_tsne(
        gap_features,
        labels,
        retinal_dataset.classes,
        dataset_display_name="Retinal",
        output_filename="tsne_retinal_gap.png",
    )

    print("Bước 7: tạo Grad-CAM retinal")
    retinal_selected_indices = select_images_per_class(retinal_dataset)
    retinal_gradcam = create_cam_outputs(
        retinal_model,
        retinal_dataset,
        retinal_selected_indices,
        device,
        cam_class=GradCAM,
    )

    print("Bước 8: tạo Grad-CAM++ retinal")
    retinal_gradcam_plus_plus = create_cam_outputs(
        retinal_model,
        retinal_dataset,
        retinal_selected_indices,
        device,
        cam_class=GradCAMPlusPlus,
    )

    print("Bước 9: ghép hình dọc Input | Grad-CAM | Grad-CAM++")
    draw_vertical_cam_comparison(
        bt_mri_dataset,
        bt_mri_gradcam,
        bt_mri_gradcam_plus_plus,
        retinal_dataset,
        retinal_gradcam,
        retinal_gradcam_plus_plus,
    )

    print("Đã lưu XAI tại:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

