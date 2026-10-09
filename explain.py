"""Create Grad-CAM, Grad-CAM++, and t-SNE figures for both datasets.

CAM methods target ``model.fusion.output_projection``, whose output has shape
``[B, 768, 7, 7]`` after Conv 1x1, BatchNorm, and ReLU. t-SNE uses the
``[B, 768]`` vector after global average pooling and before the classifier.
"""

import argparse
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
from utils.config import (
    build_model_from_config,
    load_config,
    resolve_path,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 2026
IMAGES_PER_CLASS = 4
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "xai"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Grad-CAM, Grad-CAM++ and t-SNE figures"
    )
    parser.add_argument("--bt-mri-config", required=True)
    parser.add_argument("--bt-mri-checkpoint", required=True)
    parser.add_argument("--bt-mri-data-root")
    parser.add_argument("--retinal-config", required=True)
    parser.add_argument("--retinal-checkpoint", required=True)
    parser.add_argument("--retinal-data-root")
    parser.add_argument("--output-dir", default="outputs/xai")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--images-per-class", type=int, default=4)
    return parser.parse_args()


def set_xai_seed(seed):
    """Seed image selection and dimensionality reduction without locking CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_checkpoint(checkpoint_path, dataset_name):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint not found for " + dataset_name + ": " + str(checkpoint_path)
        )
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
        raise FileNotFoundError("Test dataset not found: " + str(test_folder))

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


def create_model(config, number_of_classes, device):
    """Build the architecture used to train the checkpoint."""
    return build_model_from_config(
        config,
        number_of_classes=number_of_classes,
        device=device,
        pretrained=False,
    )


def load_model(config, checkpoint_path, classes, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_classes = checkpoint.get("classes")
    if checkpoint_classes is not None:
        if list(checkpoint_classes) != list(classes):
            raise ValueError(
                "Checkpoint class order does not match the test dataset.\n"
                + "Checkpoint: " + str(list(checkpoint_classes)) + "\n"
                + "Test data: " + str(list(classes))
            )

    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model = create_model(config, len(classes), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def forward_with_gap(model, images):
    """Run a forward pass and return the vector before the classifier."""
    images = model.encoder(images)
    feature_1 = model.cnn_1(images)
    feature_2 = model.cnn_2(images)
    fused_tokens = model.fusion(feature_1, feature_2)
    gap_features = fused_tokens.mean(dim=1)
    logits = model.classifier(gap_features)
    return logits, gap_features


@torch.no_grad()
def extract_gap_features(model, loader, device):
    """Extract [N, 768] global-average-pooled features for t-SNE."""
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
    """Apply StandardScaler and t-SNE directly to GAP features without PCA."""
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
    """Randomly select a fixed number of images per class."""
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
    """Create heatmaps from one CAM method for the selected images."""
    outputs = {}

    # output_projection returns [B, 768, 7, 7], so no reshape is required.
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
    """Draw both datasets side by side with three columns per dataset."""
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

            # Horizontal rules separate class groups.
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
    global IMAGE_SIZE
    global BATCH_SIZE
    global NUM_WORKERS
    global RANDOM_SEED
    global IMAGES_PER_CLASS
    global OUTPUT_DIR

    args = parse_args()
    bt_mri_config = load_config(args.bt_mri_config)
    retinal_config = load_config(args.retinal_config)
    validate_config(bt_mri_config)
    validate_config(retinal_config)

    if args.bt_mri_data_root:
        bt_mri_config["data_root"] = args.bt_mri_data_root
    if args.retinal_data_root:
        retinal_config["data_root"] = args.retinal_data_root

    if bt_mri_config["model"]["image_size"] != retinal_config["model"]["image_size"]:
        raise ValueError("Both configs must use the same image_size")

    IMAGE_SIZE = bt_mri_config["model"]["image_size"]
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    RANDOM_SEED = args.random_seed
    IMAGES_PER_CLASS = args.images_per_class
    OUTPUT_DIR = resolve_path(args.output_dir, PROJECT_ROOT)

    bt_mri_checkpoint_path = resolve_path(args.bt_mri_checkpoint, PROJECT_ROOT)
    retinal_checkpoint_path = resolve_path(args.retinal_checkpoint, PROJECT_ROOT)
    bt_mri_test_folder = (
        resolve_path(bt_mri_config["data_root"], PROJECT_ROOT) / "test"
    )
    retinal_test_folder = (
        resolve_path(retinal_config["data_root"], PROJECT_ROOT) / "test"
    )

    set_xai_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Step 1: load the BT-MRI checkpoint")
    check_checkpoint(bt_mri_checkpoint_path, "BT-MRI")
    bt_mri_dataset, bt_mri_loader = build_test_data(bt_mri_test_folder)
    bt_mri_model = load_model(
        bt_mri_config,
        bt_mri_checkpoint_path,
        bt_mri_dataset.classes,
        device,
    )

    print("Step 2: extract BT-MRI GAP features and create t-SNE")
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

    print("Step 3: create BT-MRI Grad-CAM")
    bt_mri_selected_indices = select_images_per_class(bt_mri_dataset)
    bt_mri_gradcam = create_cam_outputs(
        bt_mri_model,
        bt_mri_dataset,
        bt_mri_selected_indices,
        device,
        cam_class=GradCAM,
    )

    print("Step 4: create BT-MRI Grad-CAM++")
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

    print("Step 5: load the retinal checkpoint")
    check_checkpoint(retinal_checkpoint_path, "retinal")
    retinal_dataset, retinal_loader = build_test_data(retinal_test_folder)
    retinal_model = load_model(
        retinal_config,
        retinal_checkpoint_path,
        retinal_dataset.classes,
        device,
    )

    print("Step 6: extract retinal GAP features and create t-SNE")
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

    print("Step 7: create retinal Grad-CAM")
    retinal_selected_indices = select_images_per_class(retinal_dataset)
    retinal_gradcam = create_cam_outputs(
        retinal_model,
        retinal_dataset,
        retinal_selected_indices,
        device,
        cam_class=GradCAM,
    )

    print("Step 8: create retinal Grad-CAM++")
    retinal_gradcam_plus_plus = create_cam_outputs(
        retinal_model,
        retinal_dataset,
        retinal_selected_indices,
        device,
        cam_class=GradCAMPlusPlus,
    )

    print("Step 9: combine Input | Grad-CAM | Grad-CAM++ panels")
    draw_vertical_cam_comparison(
        bt_mri_dataset,
        bt_mri_gradcam,
        bt_mri_gradcam_plus_plus,
        retinal_dataset,
        retinal_gradcam,
        retinal_gradcam_plus_plus,
    )

    print("XAI outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

