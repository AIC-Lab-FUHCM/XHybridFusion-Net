import json
from pathlib import Path

import torch
import torch.nn as nn

# Data
from data import build_dataloaders, build_eval_transform, build_train_transform

# Build model
from model import build_network

# Các CNN để lựa chọn
from model.modules.cnn_backbones import (
    DenseNet121Backbone,
    EfficientNetB0Backbone,
    EfficientNetB1Backbone,
    ResNet18Backbone,
    ResNet50Backbone,
)

# Các classifier để lựa chọn
from model.modules.classifier import MLPGELUClassifier, MLPOldClassifier
from model.modules.fusion import GAMFusionNew, GAMFusionOld

# Train, test, metric, checkpoint
from utils import (
    calculate_metrics,
    evaluate,
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    set_seed,
    train_one_epoch,
)


# ================================================================
# CHỈ CẦN CHỈNH PHẦN NÀY
# ================================================================
DATASET = "retinal"                   # Tập retinal
DATA_ROOT = "/content/datasets/retinal"  # Dataset đã copy từ Drive sang Colab

BIFORMER_DEPTH = 4                    # Số BiFormer block
PATCH_SIZE = 4                        # Kích thước patch: 2, 4, 8...
N_WIN = 7                             # Số region theo mỗi chiều
TOPK = 4                              # Mỗi region chọn TOPK region liên quan
USE_PE = True                         # Thêm learnable PE sau Conv2d chia patch
DROPOUT = 0.1                         # Dropout cho DenseNet, EfficientNet và GAM
MLP_DROPOUT = 0.1                     # Dropout sau mỗi hidden layer của classifier MLP
DROP_PATH = 0.1                       # DropPath trong mỗi BiFormer block
CNN_1 = DenseNet121Backbone           # Feature 1 của GAM
CNN_2 = EfficientNetB0Backbone        # Feature 2 của GAM
# Có thể chọn: ResNet18Backbone, ResNet50Backbone,
# EfficientNetB0Backbone, EfficientNetB1Backbone, DenseNet121Backbone
# Retinal: GAM mới + MLP GELU mới, seed 42.
CLASSIFIER = MLPGELUClassifier

FUSION = GAMFusionNew

CNN_1_FREEZE_BLOCKS = 0               # Fine-tune toàn bộ DenseNet
CNN_2_FREEZE_BLOCKS = 0               # Fine-tune toàn bộ EfficientNet

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0                       # 0 để tái lập augmentation chính xác
SEED = 42                             # Seed chung cho toàn bộ lần chạy
NUM_EPOCHS = 300
USE_EARLY_STOPPING = False            # False: luôn chạy đủ NUM_EPOCHS
PATIENCE = 15                         # Dừng nếu val loss không tốt hơn 15 epoch
MIN_DELTA = 1e-4                      # Mức giảm val loss tối thiểu
# Epoch 1-50: BiFormer học nhanh hơn, hai CNN/fusion/classifier học chậm.
FIRST_STAGE_EPOCHS = 50
FIRST_STAGE_BIFORMER_LR = 1e-4
FIRST_STAGE_CNN_LR = 1e-5
FIRST_STAGE_FUSION_LR = 1e-5
FIRST_STAGE_CLASSIFIER_LR = 1e-5

# Epoch 51-300: fine-tune toàn bộ model và giảm LR bằng cosine.
SECOND_STAGE_LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4                   # Đặt 0 nếu không muốn dùng weight decay
PRETRAINED = True
RESUME_TRAINING = True                # True: tiếp tục từ epoch đã lưu
SAVE_LAST_EVERY = 25                  # Cập nhật last checkpoint mỗi 25 epoch
# ================================================================


PROJECT_ROOT = Path(__file__).resolve().parent


def set_learning_rates(optimizer, epoch):
    if epoch <= FIRST_STAGE_EPOCHS:
        learning_rates = {
            "biformer": FIRST_STAGE_BIFORMER_LR,
            "cnn_1": FIRST_STAGE_CNN_LR,
            "cnn_2": FIRST_STAGE_CNN_LR,
            "fusion": FIRST_STAGE_FUSION_LR,
            "classifier": FIRST_STAGE_CLASSIFIER_LR,
        }
    else:
        learning_rates = {
            "biformer": SECOND_STAGE_LR,
            "cnn_1": SECOND_STAGE_LR,
            "cnn_2": SECOND_STAGE_LR,
            "fusion": SECOND_STAGE_LR,
            "classifier": SECOND_STAGE_LR,
        }

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rates[parameter_group["name"]]


def learning_rate_text(optimizer):
    return ", ".join(
        group["name"] + "=" + f"{group['lr']:.1e}"
        for group in optimizer.param_groups
    )


def main():
    if PATCH_SIZE <= 0 or N_WIN <= 0:
        raise ValueError("PATCH_SIZE và N_WIN phải lớn hơn 0")
    if TOPK <= 0 or TOPK > N_WIN * N_WIN:
        raise ValueError("TOPK phải từ 1 đến N_WIN x N_WIN")
    if IMAGE_SIZE % PATCH_SIZE != 0:
        raise ValueError("IMAGE_SIZE phải chia hết cho PATCH_SIZE")
    if (IMAGE_SIZE // PATCH_SIZE) % N_WIN != 0:
        raise ValueError("Kích thước feature sau chia patch phải chia hết cho N_WIN")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bước 1: preprocessing và augmentation.
    train_transform = build_train_transform(image_size=IMAGE_SIZE)
    eval_transform = build_eval_transform(image_size=IMAGE_SIZE)

    # Bước 2: tạo train/val/test DataLoader.
    data = build_dataloaders(
        data_root=PROJECT_ROOT / DATA_ROOT,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )

    # Bước 3: BiFormer -> hai CNN -> fusion -> GAP -> classifier.
    model = build_network(
        num_classes=len(data["classes"]),
        biformer_depth=BIFORMER_DEPTH,
        patch_size=PATCH_SIZE,
        n_win=N_WIN,
        topk=TOPK,
        image_size=IMAGE_SIZE,
        use_pe=USE_PE,
        dropout=DROPOUT,
        mlp_dropout=MLP_DROPOUT,
        drop_path=DROP_PATH,
        cnn_1_class=CNN_1,
        cnn_2_class=CNN_2,
        fusion_class=FUSION,
        classifier_class=CLASSIFIER,
        fusion_dim=FUSION.default_embed_dim,
        fusion_heads=8,
        pretrained=PRETRAINED,
        cnn_1_freeze_blocks=CNN_1_FREEZE_BLOCKS,
        cnn_2_freeze_blocks=CNN_2_FREEZE_BLOCKS,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": FIRST_STAGE_BIFORMER_LR, "name": "biformer"},
            {"params": model.cnn_1.parameters(), "lr": FIRST_STAGE_CNN_LR, "name": "cnn_1"},
            {"params": model.cnn_2.parameters(), "lr": FIRST_STAGE_CNN_LR, "name": "cnn_2"},
            {"params": model.fusion.parameters(), "lr": FIRST_STAGE_FUSION_LR, "name": "fusion"},
            {"params": model.classifier.parameters(), "lr": FIRST_STAGE_CLASSIFIER_LR, "name": "classifier"},
        ],
        weight_decay=WEIGHT_DECAY
    )

    # Cosine chỉ chạy trong giai đoạn 2. Đặt base LR của mọi nhóm là 1e-4.
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = SECOND_STAGE_LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS - FIRST_STAGE_EPOCHS,
        eta_min=MIN_LR,
    )

    fusion_name = (
        FUSION.__name__
        + "_"
        + CLASSIFIER.__name__
        + "_FullDrop0p1"
        + "_TwoStageLR_Cosine300_Seed"
        + str(SEED)
    )
    print("Dataset:", DATASET, "| Fusion:", fusion_name)
    best_loss_checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / f"best_loss_{DATASET}_{fusion_name}.pt"
    )
    best_accuracy_checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / f"best_acc_{DATASET}_{fusion_name}.pt"
    )
    last_checkpoint_path = (
        PROJECT_ROOT
        / "checkpoints"
        / f"last_{DATASET}_{fusion_name}.pt"
    )

    # Bước 4: train và lưu model tốt nhất theo val loss và val accuracy.
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_accuracy = float("-inf")
    early_stop_best_loss = float("inf")
    early_stop_counter = 0

    # Nếu Colab bị ngắt, chạy lại file fusion tương ứng để tiếp tục.
    if RESUME_TRAINING and last_checkpoint_path.exists():
        checkpoint = load_training_checkpoint(
            model,
            optimizer,
            last_checkpoint_path,
            device,
            scheduler=scheduler,
            data_generator=data["train_generator"],
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_val_accuracy = checkpoint.get("best_val_accuracy", float("-inf"))
        early_stop_best_loss = checkpoint.get(
            "early_stop_best_loss",
            best_val_loss
        )
        early_stop_counter = checkpoint.get("early_stop_counter", 0)
        print("Tiếp tục train từ epoch", start_epoch)

        if not USE_EARLY_STOPPING:
            early_stop_best_loss = best_val_loss
            early_stop_counter = 0

        if USE_EARLY_STOPPING and early_stop_counter >= PATIENCE:
            print("Checkpoint đã đạt điều kiện early stopping")
            start_epoch = NUM_EPOCHS + 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Giai đoạn 1 dùng LR cố định; epoch 51 đặt LR gốc cho cosine.
        if epoch <= FIRST_STAGE_EPOCHS + 1:
            set_learning_rates(optimizer, epoch)

        train_loss, train_accuracy = train_one_epoch(
            model, data["train"], criterion, optimizer, device
        )
        val_loss, val_labels, val_predictions = evaluate(
            model, data["val"], criterion, device
        )
        val_metrics = calculate_metrics(val_labels, val_predictions)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"lr: {learning_rate_text(optimizer)}"
        )

        val_accuracy = val_metrics["accuracy"]

        # Lưu model có validation loss thấp nhất.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                path=best_loss_checkpoint_path,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                classes=data["classes"],
            )

        # Lưu model có validation accuracy cao nhất.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model=model,
                path=best_accuracy_checkpoint_path,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                classes=data["classes"],
            )

        # Early stopping vẫn theo validation loss.
        if USE_EARLY_STOPPING:
            if val_loss < early_stop_best_loss - MIN_DELTA:
                early_stop_best_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(
                    f"Early stopping: {early_stop_counter}/{PATIENCE} "
                    "epoch không cải thiện"
                )
        else:
            early_stop_best_loss = min(early_stop_best_loss, val_loss)
            early_stop_counter = 0

        # Giảm LR sau khi hoàn thành mỗi epoch của giai đoạn 2.
        if epoch > FIRST_STAGE_EPOCHS:
            scheduler.step()

        # Giảm số lần ghi file checkpoint lớn lên Google Drive.
        should_save_last = (
            epoch % SAVE_LAST_EVERY == 0
            or epoch == NUM_EPOCHS
            or (
                USE_EARLY_STOPPING
                and early_stop_counter >= PATIENCE
            )
        )

        if should_save_last:
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                path=last_checkpoint_path,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_val_accuracy=best_val_accuracy,
                early_stop_best_loss=early_stop_best_loss,
                classes=data["classes"],
                early_stop_counter=early_stop_counter,
                scheduler=scheduler,
                data_generator=data["train_generator"],
            )

        if USE_EARLY_STOPPING and early_stop_counter >= PATIENCE:
            print("Dừng sớm tại epoch", epoch)
            break

    # Bước 5: test model có validation loss thấp nhất.
    loss_checkpoint = load_checkpoint(
        model,
        best_loss_checkpoint_path,
        device
    )
    test_loss, test_labels, test_predictions = evaluate(
        model, data["test"], criterion, device
    )
    loss_test_metrics = calculate_metrics(test_labels, test_predictions)
    loss_test_metrics["loss"] = test_loss
    loss_test_metrics["checkpoint_epoch"] = loss_checkpoint["epoch"]
    loss_test_metrics["validation_loss"] = loss_checkpoint["val_loss"]
    loss_test_metrics["validation_accuracy"] = loss_checkpoint["val_accuracy"]
    loss_test_metrics["fusion"] = fusion_name

    print("\nTest model có validation loss thấp nhất:")
    print(json.dumps(loss_test_metrics, indent=2))

    loss_results_path = (
        PROJECT_ROOT
        / "results"
        / f"metrics_best_loss_{DATASET}_{fusion_name}.json"
    )
    loss_results_path.parent.mkdir(parents=True, exist_ok=True)
    with loss_results_path.open("w", encoding="utf-8") as file:
        json.dump(loss_test_metrics, file, indent=2)

    # Bước 6: test model có validation accuracy cao nhất.
    accuracy_checkpoint = load_checkpoint(
        model,
        best_accuracy_checkpoint_path,
        device
    )
    test_loss, test_labels, test_predictions = evaluate(
        model, data["test"], criterion, device
    )
    accuracy_test_metrics = calculate_metrics(test_labels, test_predictions)
    accuracy_test_metrics["loss"] = test_loss
    accuracy_test_metrics["checkpoint_epoch"] = accuracy_checkpoint["epoch"]
    accuracy_test_metrics["validation_loss"] = accuracy_checkpoint["val_loss"]
    accuracy_test_metrics["validation_accuracy"] = (
        accuracy_checkpoint["val_accuracy"]
    )
    accuracy_test_metrics["fusion"] = fusion_name

    print("\nTest model có validation accuracy cao nhất:")
    print(json.dumps(accuracy_test_metrics, indent=2))

    accuracy_results_path = (
        PROJECT_ROOT
        / "results"
        / f"metrics_best_acc_{DATASET}_{fusion_name}.json"
    )
    with accuracy_results_path.open("w", encoding="utf-8") as file:
        json.dump(accuracy_test_metrics, file, indent=2)


if __name__ == "__main__":
    main()





