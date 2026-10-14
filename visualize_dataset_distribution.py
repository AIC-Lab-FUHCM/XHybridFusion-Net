"""Count dataset images and plot train/validation/test distributions."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot train, validation, and test image counts"
    )
    parser.add_argument("--retinal-root", required=True)
    parser.add_argument("--bt-mri-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/dataset_statistics",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def count_images(folder):
    if not folder.exists():
        return 0
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_counts(dataset_root):
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    for split in SPLITS:
        split_folder = dataset_root / split
        if not split_folder.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_folder}")

    class_names = sorted(
        {
            class_folder.name
            for split in SPLITS
            for class_folder in (dataset_root / split).iterdir()
            if class_folder.is_dir()
        }
    )
    if not class_names:
        raise ValueError(f"No class directories found in: {dataset_root}")

    rows = []
    for class_name in class_names:
        row = {"class": class_name}
        for split in SPLITS:
            row[split] = count_images(dataset_root / split / class_name)
        row["total"] = sum(row[split] for split in SPLITS)
        rows.append(row)
    return rows


def write_csv(all_results, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["dataset", "class", "train", "val", "test", "total"],
        )
        writer.writeheader()
        for dataset_name, rows in all_results.items():
            for row in rows:
                writer.writerow({"dataset": dataset_name, **row})


def display_class_name(class_name):
    names = {
        "notumor": "No Tumor",
        "diabetic_retinopathy": "Diabetic Retinopathy",
    }
    return names.get(class_name, class_name.replace("_", " ").title())


def draw_table(axis, dataset_name, rows):
    axis.axis("off")
    totals = {
        split: sum(row[split] for row in rows)
        for split in SPLITS
    }
    grand_total = sum(totals.values())

    table_rows = [
        [
            display_class_name(row["class"]),
            f"{row['train']:,}",
            f"{row['val']:,}",
            f"{row['test']:,}",
            f"{row['total']:,}",
        ]
        for row in rows
    ]
    table_rows.append(
        [
            "Total Images",
            f"{totals['train']:,}",
            f"{totals['val']:,}",
            f"{totals['test']:,}",
            f"{grand_total:,}",
        ]
    )

    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "Class",
            "Training Images",
            "Validation Images",
            "Testing Images",
            "Total",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.30, 0.19, 0.21, 0.19, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.75)

    for column in range(5):
        header = table[(0, column)]
        header.set_facecolor("#D9E6F2")
        header.set_text_props(weight="bold")

        total_cell = table[(len(table_rows), column)]
        total_cell.set_facecolor("#EAEAEA")
        total_cell.set_text_props(weight="bold")

    axis.set_title(dataset_name, fontsize=13, fontweight="bold", pad=14)


def draw_figure(all_results, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    draw_table(axes[0], "Retinal Dataset", all_results["retinal"])
    draw_table(axes[1], "BT-MRI Dataset", all_results["bt_mri"])
    figure.suptitle("Dataset Statistics by Class", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def print_counts(dataset_name, rows):
    totals = {
        split: sum(row[split] for row in rows)
        for split in SPLITS
    }
    total = sum(totals.values())
    print(
        f"{dataset_name}: train={totals['train']}, val={totals['val']}, "
        f"test={totals['test']}, total={total}"
    )


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "retinal": collect_counts(resolve_path(args.retinal_root)),
        "bt_mri": collect_counts(resolve_path(args.bt_mri_root)),
    }

    write_csv(all_results, output_dir / "dataset_split_counts.csv")
    draw_figure(all_results, output_dir / "dataset_split_table.png")

    for dataset_name, rows in all_results.items():
        print_counts(dataset_name, rows)
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
