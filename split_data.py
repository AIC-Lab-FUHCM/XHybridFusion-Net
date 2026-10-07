from pathlib import Path
import random
import shutil


# ================================================================
# CHỈ CẦN CHỈNH PHẦN NÀY
# ================================================================
DATASET = "bt_mri_hctnet"
# "retinal", "bt_mri_effresnet", "bt_mri_hctnet" hoặc "all"

# Seed giúp mỗi lần chia đều chọn đúng các ảnh giống nhau.
RETINAL_SEED = 4242
BT_MRI_EFFRESNET_SEED = 142
BT_MRI_HCTNET_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent

# Thư mục dữ liệu gốc sau khi giải nén.
RETINAL_SOURCE = PROJECT_ROOT / "raw_data/retinal/dataset"
BT_MRI_SOURCE = PROJECT_ROOT / "raw_data/bt_mri"

# Thư mục sau khi chia, main.py sẽ đọc dữ liệu tại đây.
RETINAL_OUTPUT = PROJECT_ROOT / "datasets/retinal"
BT_MRI_EFFRESNET_OUTPUT = PROJECT_ROOT / "datasets/bt_mri_effresnet"
BT_MRI_HCTNET_OUTPUT = PROJECT_ROOT / "datasets/bt_mri_hctnet"
# ================================================================


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_images(folder):
    images = []

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)

    return sorted(images)


def check_output_folder(output_folder):
    # Không ghi đè nếu thư mục đích đã có ảnh.
    if output_folder.exists():
        old_images = []

        for file in output_folder.rglob("*"):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                old_images.append(file)

        if len(old_images) > 0:
            raise ValueError(
                "Thư mục đích đã có ảnh: " + str(output_folder)
            )


def copy_images(images, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)

    for image in images:
        shutil.copy2(image, output_folder / image.name)


def get_retinal_group(image):
    name = image.stem

    # Ví dụ: _129_3024383 và _129_5296959 cùng ảnh nguồn 129.
    if name.startswith("_"):
        return name.split("_")[1]

    # Ví dụ: 10003_left và 10003_right cùng bệnh nhân 10003.
    if name.endswith("_left") or name.endswith("_right"):
        return name.rsplit("_", 1)[0]

    # Ảnh không có cặp thì dùng toàn bộ tên làm mã riêng.
    return name


def split_retinal(source_folder, output_folder, seed=4242):
    """Chia Retinal gần 70% train, 15% val và 15% test."""
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    if not source_folder.exists():
        raise ValueError("Không tìm thấy Retinal gốc: " + str(source_folder))

    check_output_folder(output_folder)
    random_generator = random.Random(seed)

    for class_folder in sorted(source_folder.iterdir()):
        if not class_folder.is_dir():
            continue

        # Gom ảnh cùng bệnh nhân hoặc cùng ảnh nguồn vào chung một nhóm.
        groups = {}

        for image in get_images(class_folder):
            group_name = get_retinal_group(image)

            if group_name not in groups:
                groups[group_name] = []

            groups[group_name].append(image)

        group_list = list(groups.values())
        random_generator.shuffle(group_list)

        number_of_groups = len(group_list)
        # Chia theo nhóm bệnh nhân: 70% train, 15% val và 15% test.
        train_end = int(number_of_groups * 0.70)
        val_end = train_end + int(number_of_groups * 0.15)

        train_groups = group_list[:train_end]
        val_groups = group_list[train_end:val_end]
        test_groups = group_list[val_end:]

        train_images = []
        val_images = []
        test_images = []

        for group in train_groups:
            train_images.extend(group)

        for group in val_groups:
            val_images.extend(group)

        for group in test_groups:
            test_images.extend(group)

        copy_images(
            train_images,
            output_folder / "train" / class_folder.name
        )
        copy_images(
            val_images,
            output_folder / "val" / class_folder.name
        )
        copy_images(
            test_images,
            output_folder / "test" / class_folder.name
        )

        print(class_folder.name)
        print("  train:", len(train_images))
        print("  val:  ", len(val_images))
        print("  test: ", len(test_images))

    source_text = (
        "Dataset: Eye Diseases Classification\n"
        "Source: https://www.kaggle.com/datasets/"
        "gunavenkatdoddi/eye-diseases-classification\n"
        "Split: 70% train, 15% validation, 15% test\n"
        "Seed: " + str(seed) + "\n"
        "Related source/patient images stay in the same split.\n"
    )
    (output_folder / "SOURCE.txt").write_text(
        source_text,
        encoding="utf-8"
    )


def split_bt_mri_effresnet(source_folder, output_folder, seed=142):
    """EFFResNet-ViT: tách 20% Training làm val, giữ nguyên test."""
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)
    training_folder = source_folder / "Training"
    testing_folder = source_folder / "Testing"

    if not training_folder.exists() or not testing_folder.exists():
        raise ValueError("Không tìm thấy Training/Testing của BT-MRI")

    check_output_folder(output_folder)
    random_generator = random.Random(seed)

    for class_folder in sorted(training_folder.iterdir()):
        if not class_folder.is_dir():
            continue

        training_images = get_images(class_folder)
        random_generator.shuffle(training_images)

        # Tỉ lệ theo bài báo: 80% train và 20% validation.
        number_of_val_images = int(len(training_images) * 0.20)
        val_images = training_images[:number_of_val_images]
        train_images = training_images[number_of_val_images:]

        official_test_folder = testing_folder / class_folder.name
        test_images = get_images(official_test_folder)

        copy_images(
            train_images,
            output_folder / "train" / class_folder.name
        )
        copy_images(
            val_images,
            output_folder / "val" / class_folder.name
        )
        copy_images(
            test_images,
            output_folder / "test" / class_folder.name
        )

        print(class_folder.name)
        print("  train:", len(train_images))
        print("  val:  ", len(val_images))
        print("  test: ", len(test_images))

    source_text = (
        "Dataset: Brain Tumor MRI Dataset, Kaggle version 1\n"
        "Protocol: EFFResNet-ViT, IEEE Access 2025\n"
        "Source: https://www.kaggle.com/datasets/"
        "masoudnickparvar/brain-tumor-mri-dataset/versions/1\n"
        "Total: 7023 images\n"
        "Original Training: 80% train, 20% validation\n"
        "Original Testing: kept unchanged as test\n"
        "Seed: " + str(seed) + "\n"
    )
    (output_folder / "SOURCE.txt").write_text(
        source_text,
        encoding="utf-8"
    )


def split_bt_mri_hctnet(source_folder, output_folder, seed=42):
    """HCTNet: chia Training 85/15 có stratify, giữ nguyên test."""
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)
    training_folder = source_folder / "Training"
    testing_folder = source_folder / "Testing"

    if not training_folder.exists() or not testing_folder.exists():
        raise ValueError("Không tìm thấy Training/Testing của BT-MRI")

    check_output_folder(output_folder)

    # Chia riêng trong từng lớp để giữ tỉ lệ lớp (stratified split).
    random_generator = random.Random(seed)

    for class_folder in sorted(training_folder.iterdir()):
        if not class_folder.is_dir():
            continue

        training_images = get_images(class_folder)
        random_generator.shuffle(training_images)

        # round theo từng lớp cho đúng tổng của HCTNet: 4.855/857.
        number_of_val_images = round(len(training_images) * 0.15)
        val_images = training_images[:number_of_val_images]
        train_images = training_images[number_of_val_images:]

        official_test_folder = testing_folder / class_folder.name
        test_images = get_images(official_test_folder)

        copy_images(
            train_images,
            output_folder / "train" / class_folder.name
        )
        copy_images(
            val_images,
            output_folder / "val" / class_folder.name
        )
        copy_images(
            test_images,
            output_folder / "test" / class_folder.name
        )

        print(class_folder.name)
        print("  train:", len(train_images))
        print("  val:  ", len(val_images))
        print("  test: ", len(test_images))

    source_text = (
        "Dataset: Brain Tumor MRI Dataset, Kaggle version 1\n"
        "Protocol: HCTNet 2026\n"
        "Source: https://www.kaggle.com/datasets/"
        "masoudnickparvar/brain-tumor-mri-dataset/versions/1\n"
        "Total: 7023 images\n"
        "Original Training: stratified 85% train, 15% validation\n"
        "Original Testing: kept unchanged as test\n"
        "Seed: " + str(seed) + "\n"
    )
    (output_folder / "SOURCE.txt").write_text(
        source_text,
        encoding="utf-8"
    )


if __name__ == "__main__":
    if DATASET == "retinal" or DATASET == "all":
        print("\nChia Retinal")
        split_retinal(
            RETINAL_SOURCE,
            RETINAL_OUTPUT,
            RETINAL_SEED
        )

    if DATASET == "bt_mri_effresnet" or DATASET == "all":
        print("\nChia BT-MRI theo EFFResNet-ViT")
        split_bt_mri_effresnet(
            BT_MRI_SOURCE,
            BT_MRI_EFFRESNET_OUTPUT,
            BT_MRI_EFFRESNET_SEED
        )

    if DATASET == "bt_mri_hctnet" or DATASET == "all":
        print("\nChia BT-MRI theo HCTNet")
        split_bt_mri_hctnet(
            BT_MRI_SOURCE,
            BT_MRI_HCTNET_OUTPUT,
            BT_MRI_HCTNET_SEED
        )

    valid_datasets = {
        "retinal",
        "bt_mri_effresnet",
        "bt_mri_hctnet",
        "all",
    }
    if DATASET not in valid_datasets:
        raise ValueError(
            "DATASET phải là 'retinal', 'bt_mri_effresnet', "
            "'bt_mri_hctnet' hoặc 'all'"
        )
