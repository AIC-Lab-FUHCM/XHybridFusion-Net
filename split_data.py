import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split retinal or BT-MRI dataset")
    parser.add_argument("--dataset", required=True, choices=("retinal", "bt_mri"))
    parser.add_argument("--source", required=True, help="Original dataset directory")
    parser.add_argument("--output", required=True, help="Output train/val/test directory")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_images(folder):
    return sorted(
        path
        for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_empty_output(output_folder):
    output_folder = Path(output_folder)
    if output_folder.exists():
        old_images = [
            path
            for path in output_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if old_images:
            raise ValueError(
                "The output directory already contains images. Use an empty directory: "
                + str(output_folder)
            )


def copy_images(images, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(image, output_folder / image.name)


def retinal_group_id(image):
    name = image.stem
    if name.startswith("_"):
        parts = name.split("_")
        if len(parts) > 1:
            return parts[1]
    if name.endswith("_left") or name.endswith("_right"):
        return name.rsplit("_", 1)[0]
    return name


def split_retinal(source, output, seed):
    """Split retinal source/patient groups into 70/15/15 partitions."""
    source = Path(source)
    output = Path(output)
    if not source.exists():
        raise FileNotFoundError("Retinal source directory not found: " + str(source))

    ensure_empty_output(output)
    generator = random.Random(seed)

    for class_folder in sorted(path for path in source.iterdir() if path.is_dir()):
        groups = {}
        for image in get_images(class_folder):
            groups.setdefault(retinal_group_id(image), []).append(image)

        grouped_images = list(groups.values())
        generator.shuffle(grouped_images)
        train_end = int(len(grouped_images) * 0.70)
        val_end = train_end + int(len(grouped_images) * 0.15)
        split_groups = {
            "train": grouped_images[:train_end],
            "val": grouped_images[train_end:val_end],
            "test": grouped_images[val_end:],
        }

        counts = {}
        for split_name, groups_for_split in split_groups.items():
            images = [image for group in groups_for_split for image in group]
            copy_images(images, output / split_name / class_folder.name)
            counts[split_name] = len(images)

        print(class_folder.name, counts)

    (output / "SOURCE.txt").write_text(
        "Dataset: Eye Diseases Classification\n"
        "Source: https://www.kaggle.com/datasets/"
        "gunavenkatdoddi/eye-diseases-classification\n"
        "Split: 70% train, 15% validation, 15% test\n"
        f"Seed: {seed}\n"
        "Related source/patient images stay in the same split.\n",
        encoding="utf-8",
    )


def split_bt_mri(source, output, seed):
    """Split Training 80/20 and preserve the official Testing directory."""
    source = Path(source)
    output = Path(output)
    training_folder = source / "Training"
    testing_folder = source / "Testing"
    if not training_folder.exists() or not testing_folder.exists():
        raise FileNotFoundError(
            "The BT-MRI source must contain Training and Testing directories"
        )

    ensure_empty_output(output)
    generator = random.Random(seed)

    for class_folder in sorted(path for path in training_folder.iterdir() if path.is_dir()):
        images = get_images(class_folder)
        generator.shuffle(images)
        number_of_validation_images = int(len(images) * 0.20)
        validation_images = images[:number_of_validation_images]
        training_images = images[number_of_validation_images:]
        testing_images = get_images(testing_folder / class_folder.name)

        copy_images(training_images, output / "train" / class_folder.name)
        copy_images(validation_images, output / "val" / class_folder.name)
        copy_images(testing_images, output / "test" / class_folder.name)
        print(
            class_folder.name,
            {
                "train": len(training_images),
                "val": len(validation_images),
                "test": len(testing_images),
            },
        )

    (output / "SOURCE.txt").write_text(
        "Dataset: Brain Tumor MRI Dataset, Kaggle version 1\n"
        "Source: https://www.kaggle.com/datasets/"
        "masoudnickparvar/brain-tumor-mri-dataset/versions/1\n"
        "Original Training: 80% train, 20% validation\n"
        "Original Testing: kept unchanged as test\n"
        f"Seed: {seed}\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.dataset == "retinal":
        split_retinal(args.source, args.output, args.seed)
    else:
        split_bt_mri(args.source, args.output, args.seed)


if __name__ == "__main__":
    main()
