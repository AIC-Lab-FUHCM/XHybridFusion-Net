"""Visualize intermediate feature maps from the final GAM model.

The script follows the current model exactly and captures:
1. BiFormer blocks 2 and 4.
2. DenseNet121 DenseBlocks 2 and 4.
3. EfficientNet-B0 stage 4 and final output.
4. GAM output after Conv2d(2304, 768, 1) + BatchNorm + ReLU.

For display, the script selects channels with the largest mean absolute
activation for each sample and labels them.
This is only a visualization choice; it does not change the model.

Dataset and checkpoint paths are supplied through command-line arguments, so
the script works in local, Colab, and other environments without source edits.
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch
from torchvision.datasets import ImageFolder

from data import build_eval_transform
from utils.config import (
    build_model_from_config,
    load_config,
    resolve_path,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize intermediate feature maps for BT-MRI and retinal"
    )
    parser.add_argument("--bt-mri-config", required=True)
    parser.add_argument("--bt-mri-checkpoint", required=True)
    parser.add_argument("--bt-mri-data-root", help="Override BT-MRI data_root")
    parser.add_argument("--retinal-config", required=True)
    parser.add_argument("--retinal-checkpoint", required=True)
    parser.add_argument("--retinal-data-root", help="Override retinal data_root")
    parser.add_argument(
        "--output-dir",
        default="results/intermediate_feature_maps",
    )
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--images-per-class", type=int, default=4)
    parser.add_argument("--channels", type=int, default=6)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config, num_classes, device):
    return build_model_from_config(
        config,
        number_of_classes=num_classes,
        device=device,
        pretrained=False,
    )


def load_model(config, checkpoint_path, classes, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    saved_classes = checkpoint.get("classes")
    if saved_classes is not None and list(saved_classes) != list(classes):
        raise ValueError(
            "Checkpoint class order does not match the test dataset.\n"
            f"Checkpoint: {list(saved_classes)}\n"
            f"Dataset: {list(classes)}"
        )

    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model = build_model(config, len(classes), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def biformer_outputs(model, images):
    """Return Block 2, Block 4 internal, and reconstructed BiFormer outputs."""
    encoder = model.encoder
    x = encoder.patch_embed(images)
    if encoder.pos_embed is not None:
        x = x + encoder.pos_embed

    middle_output = None
    for block_index, block in enumerate(encoder.blocks):
        x = block(x)
        if block_index == 1:  # Block 2 of the configured four blocks.
            middle_output = x.clone()

    block4_output = x.clone()  # [B,48,56,56], immediately after Block 4.
    x = encoder.norm(x)

    batch_size, _, height, width = x.shape
    x = x.reshape(
        batch_size,
        3,
        encoder.patch_size,
        encoder.patch_size,
        height,
        width,
    )
    x = x.permute(0, 1, 4, 2, 5, 3)
    final_output = x.reshape(
        batch_size,
        3,
        height * encoder.patch_size,
        width * encoder.patch_size,
    )
    return middle_output, block4_output, final_output


def gam_after_relu(model, feature_1, feature_2):
    """Return GAM's [B,768,7,7] tensor immediately after its ReLU."""
    fusion = model.fusion
    x = torch.cat([feature_1, feature_2], dim=1)

    channel_weights = x.permute(0, 2, 3, 1)
    channel_weights = fusion.channel_attention(channel_weights)
    channel_weights = channel_weights.permute(0, 3, 1, 2)
    x = x * channel_weights

    spatial_weights = fusion.spatial_attention(x)
    x = x * spatial_weights

    # output_projection = Conv2d -> BatchNorm2d -> ReLU.
    return fusion.output_projection(x)


@torch.no_grad()
def extract_features(model, images):
    """Return middle and final tensors requested for the visualization."""
    bi_middle, bi_block4, bi_output = biformer_outputs(model, images)

    dense_middle = {}
    efficient_middle = {}

    def save_dense_middle(_module, _inputs, output):
        dense_middle["value"] = output

    def save_efficient_middle(_module, _inputs, output):
        efficient_middle["value"] = output

    dense_hook = model.cnn_1.features[0].denseblock2.register_forward_hook(
        save_dense_middle
    )
    efficient_hook = model.cnn_2.features[4].register_forward_hook(
        save_efficient_middle
    )
    feature_1 = model.cnn_1(bi_output)
    feature_2 = model.cnn_2(bi_output)
    dense_hook.remove()
    efficient_hook.remove()
    gam_output = gam_after_relu(model, feature_1, feature_2)
    return {
        "BiFormer block 2 [B,48,56,56]": bi_middle,
        "BiFormer block 4 [B,48,56,56]": bi_block4,
        "DenseNet DenseBlock2 [B,512,28,28]": dense_middle["value"],
        "DenseNet end DenseBlock4 [B,1024,7,7]": feature_1,
        "EfficientNet features[4] [B,80,14,14]": efficient_middle["value"],
        "EfficientNet end features [B,1280,7,7]": feature_2,
        "GAM after ReLU [B,768,7,7]": gam_output,
    }


def normalize_map(feature_map):
    feature_map = feature_map.astype(np.float32)
    low, high = np.nanmin(feature_map), np.nanmax(feature_map)
    if high <= low:
        return np.zeros_like(feature_map)
    return (feature_map - low) / (high - low)


def selected_channels(tensor, count):
    """Select the most active channels using spatial mean activation."""
    scores = tensor.abs().mean(dim=(-1, -2))
    count = min(count, tensor.shape[0])
    return torch.topk(scores, k=count).indices.tolist()


def display_name(class_name):
    names = {
        "glioma": "Glioma Tumor",
        "meningioma": "Meningioma Tumor",
        "pituitary": "Pituitary Tumor",
        "notumor": "No Tumor",
        "cataract": "Cataract",
        "diabetic_retinopathy": "Diabetic Retinopathy",
        "glaucoma": "Glaucoma",
        "normal": "Normal",
    }
    return names.get(class_name.lower(), class_name.replace("_", " ").title())


def panel_samples(dataset_name, dataset, images_per_class, random_seed):
    """Choose three disease examples, matching the paper's three columns."""
    preferred = {
        "BT-MRI": ["glioma", "meningioma", "pituitary"],
        "Retinal": ["cataract", "diabetic_retinopathy", "glaucoma"],
    }[dataset_name]
    selected = selected_indices(dataset, images_per_class, random_seed)
    selected_by_class = {}
    for class_name, sample_number, index in selected:
        selected_by_class.setdefault(class_name.lower(), (class_name, sample_number, index))

    panels = []
    for class_name in preferred:
        if class_name in selected_by_class:
            panels.append(selected_by_class[class_name])

    # Fallback for a differently named/custom dataset: use the first three classes.
    if len(panels) < 3:
        used = {item[2] for item in panels}
        panels.extend(item for item in selected if item[2] not in used)
        panels = panels[:3]
    return panels


def draw_dataset_figure(
    dataset_name,
    samples,
    output_path,
    channel_count,
):
    """Draw one paper-like multi-row figure for one dataset."""
    row_labels = [
        "(a) BiFormer B2 | C=48",
        "(b) BiFormer B4 | C=48",
        "(c) DenseBlock2 | C=512",
        "(d) DenseBlock4 | C=1024",
        "(e) EfficientNet S4 | C=80",
        "(f) EfficientNet final | C=1280",
        "(g) GAM after ReLU | C=768",
    ]
    n_columns = len(samples)
    n_rows = len(samples[0][3])
    left_margin = 0.075
    right_margin = 0.985
    top_margin = 0.90
    bottom_margin = 0.055
    figure = plt.figure(figsize=(n_columns * 4.15, max(11.0, n_rows * 2.7)))
    outer = figure.add_gridspec(
        n_rows,
        n_columns,
        left=left_margin,
        right=right_margin,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.08,
        hspace=0.16,
    )

    for column, (class_name, sample_number, _, features, input_image) in enumerate(samples):
        figure.text(
            left_margin + (column + 0.5) * (right_margin - left_margin) / n_columns,
            0.935,
            display_name(class_name),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        for row, (title, tensor) in enumerate(features.items()):
            cell = outer[row, column].subgridspec(
                2,
                5,
                width_ratios=[0.78, 1.0, 1.0, 1.0, 1.0],
                wspace=0.035,
                hspace=0.035,
            )
            input_axis = figure.add_subplot(cell[:, 0])
            # Show the corresponding input thumbnail in every feature row,
            # as in the paper's feature visualization layout.
            input_axis.imshow(input_image)
            input_axis.set_title("Input", fontsize=7, pad=2)
            input_axis.set_xticks([])
            input_axis.set_yticks([])

            tensor = tensor[0].detach().cpu()
            map_axes = []
            if tensor.shape[0] == 3:
                # BiFormer output: visualize RGB reconstruction and its channels.
                rgb = np.stack(
                    [normalize_map(tensor[channel].numpy()) for channel in range(3)],
                    axis=-1,
                )
                axis = figure.add_subplot(cell[:, 1])
                axis.imshow(rgb)
                axis.set_title("RGB", fontsize=7, pad=2)
                map_axes.append(axis)
                for map_column, channel_id in enumerate(range(3), start=2):
                    axis = figure.add_subplot(cell[:, map_column])
                    axis.imshow(
                        normalize_map(tensor[channel_id].numpy()),
                        cmap="viridis",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    axis.set_title(f"ch {channel_id}", fontsize=7, pad=2)
                    map_axes.append(axis)
            else:
                channel_ids = selected_channels(tensor, channel_count)
                for position, channel_id in enumerate(channel_ids):
                    map_row, map_column = divmod(position, 3)
                    axis = figure.add_subplot(cell[map_row, map_column + 1])
                    axis.imshow(
                        normalize_map(tensor[channel_id].numpy()),
                        cmap="viridis",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    axis.set_title(f"ch {channel_id}", fontsize=7, pad=2)
                    map_axes.append(axis)

            for axis in map_axes:
                axis.set_xticks([])
                axis.set_yticks([])
            # A compact colorbar for each sample panel, matching the paper style.
            if map_axes:
                figure.colorbar(
                    ScalarMappable(
                        norm=Normalize(vmin=0.0, vmax=1.0),
                        cmap="viridis",
                    ),
                    ax=map_axes,
                    fraction=0.035,
                    pad=0.012,
                    ticks=[0.0, 0.5, 1.0],
                )
            figure.text(
                0.028,
                top_margin
                - (row + 0.5) * (top_margin - bottom_margin) / n_rows,
                row_labels[row],
                rotation=90,
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.suptitle(
        f"Intermediate feature maps - {dataset_name}",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def selected_indices(dataset, images_per_class, random_seed):
    """Select images exactly like select_images_per_class in the XAI script."""
    generator = random.Random(random_seed)
    result = []
    for class_id, class_name in enumerate(dataset.classes):
        class_indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_id
        ]
        count = min(images_per_class, len(class_indices))
        chosen_indices = generator.sample(class_indices, count)
        for sample_number, index in enumerate(chosen_indices, start=1):
            result.append((class_name, sample_number, index))
    return result


def process_dataset(
    dataset_name,
    config,
    checkpoint_path,
    test_root,
    output_dir,
    device,
    images_per_class,
    channel_count,
    random_seed,
):
    if not test_root.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_root}")

    dataset = ImageFolder(
        test_root,
        transform=build_eval_transform(image_size=config["model"]["image_size"]),
    )
    model = load_model(config, checkpoint_path, dataset.classes, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for class_name, sample_number, index in panel_samples(
        dataset_name,
        dataset,
        images_per_class,
        random_seed,
    ):
        image, _ = dataset[index]
        image_batch = image.unsqueeze(0).to(device)
        features = extract_features(model, image_batch)
        input_image = np.transpose(image.numpy(), (1, 2, 0))
        input_image = np.clip(input_image, 0.0, 1.0)
        samples.append((class_name, sample_number, index, features, input_image))

    output_path = output_dir / f"{dataset_name.lower()}_feature_maps.png"
    draw_dataset_figure(dataset_name, samples, output_path, channel_count)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    bt_mri_config = load_config(args.bt_mri_config)
    retinal_config = load_config(args.retinal_config)
    validate_config(bt_mri_config)
    validate_config(retinal_config)

    if args.bt_mri_data_root:
        bt_mri_config["data_root"] = args.bt_mri_data_root
    if args.retinal_data_root:
        retinal_config["data_root"] = args.retinal_data_root

    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    bt_mri_checkpoint = resolve_path(args.bt_mri_checkpoint, PROJECT_ROOT)
    retinal_checkpoint = resolve_path(args.retinal_checkpoint, PROJECT_ROOT)
    bt_mri_test_root = resolve_path(
        bt_mri_config["data_root"], PROJECT_ROOT
    ) / "test"
    retinal_test_root = resolve_path(
        retinal_config["data_root"], PROJECT_ROOT
    ) / "test"

    process_dataset(
        "BT-MRI",
        bt_mri_config,
        bt_mri_checkpoint,
        bt_mri_test_root,
        output_dir,
        device,
        args.images_per_class,
        args.channels,
        args.random_seed,
    )
    process_dataset(
        "Retinal",
        retinal_config,
        retinal_checkpoint,
        retinal_test_root,
        output_dir,
        device,
        args.images_per_class,
        args.channels,
        args.random_seed,
    )


if __name__ == "__main__":
    main()
