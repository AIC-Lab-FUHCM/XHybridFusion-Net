# XHybridFusion-Net: Attention-Guided BiFormer and Dual-CNN Learning for Explainable Medical Image Classification

PyTorch implementation of XHybridFusion-Net for four-class Brain Tumor MRI (BT-MRI) and Retinal Eye Diseases (RED) classification.

## Model

```text
Input -> Patch Embedding + BiFormer -> DenseNet-121 + EfficientNet-B0
      -> GAM Fusion -> Global Average Pooling -> GELU MLP -> Prediction
```

| Component | Configuration |
|---|---|
| Input | RGB, `224 x 224` |
| BiFormer | 4 blocks, patch size 4, 4 heads, `n_win=7`, `topk=4` |
| CNN branches | ImageNet-pretrained DenseNet-121 and EfficientNet-B0 |
| Fusion | GAM channel-spatial attention with `1 x 1` projection to 768 channels |
| Classifier | `768 -> 1024 -> 512 -> 4`, GELU |
| Dropout / drop path | 0.1 / 0.1 |
| Optimizer | AdamW, weight decay `1e-4` |
| Batch size / epochs | 32 / 300 |

## Project structure

```text
configs/                 Experiment configurations for each dataset and seed
checkpoints/             Google Drive link for pretrained weights
data/                    Transforms and dataset loading
model/                   BiFormer encoder, CNN branches, GAM, and classifier
utils/                   Training, metrics, checkpoint, and configuration utilities
third_party/BiFormer/    Included BiFormer source
results/                 Reported tables and figures
main.py                  Training entry point
evaluate.py              Test-set evaluation
explain.py               Grad-CAM, Grad-CAM++, and t-SNE
split_data.py            Dataset partitioning
```

## Installation

```bash
git clone https://github.com/thoingu68-ui/XHybridFusion-Net.git
cd XHybridFusion-Net
python -m pip install -r requirements.txt
```

Experiments used PyTorch `2.11.0+cu130`, CUDA `13.0`, and an NVIDIA A100-SXM4 40 GB GPU.

## Data preparation

Download the raw datasets from Kaggle:

- [Retinal Eye Diseases](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification): cataract, diabetic retinopathy, glaucoma, normal.
- [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1): glioma, meningioma, no tumor, pituitary.

The downloaded data are raw data, not the final `train/val/test` folders. Prepare them as follows:

```text
raw/retinal/<class>/
raw/bt_mri/Training/<class>/
raw/bt_mri/Testing/<class>/
```

Then create the experiment splits:

```bash
python split_data.py --dataset retinal --source raw/retinal --output datasets/retinal --seed 4242
python split_data.py --dataset bt_mri --source raw/bt_mri --output datasets/bt_mri --seed 142
```

```text
datasets/<dataset>/
|-- train/<class>/
|-- val/<class>/
`-- test/<class>/
```

| Dataset | Split protocol | Final split |
|---|---|---|
| RED | 70% train, 15% validation, 15% test | 2,947 / 632 / 638 (4,217 total) |
| BT-MRI | Official test retained; original training split 80/20 into train/validation | 4,571 / 1,141 / 1,311 (7,023 total) |

Training uses RGB conversion, resize to `224 x 224`, ColorJitter (brightness `0-0.6`, contrast `0-1.0`, saturation `0-0.3`, hue `0-0.1`), horizontal/vertical flip (`p=0.5`), and `-90°`/`+90°` rotation (`p=0.5`). Validation and testing only resize and convert to tensors in `[0, 1]`; no mean/std normalization is applied.

## Checkpoints

Model weights are excluded from GitHub. The Google Drive folder link is in [checkpoints/README.md](checkpoints/README.md).

## Training

```bash
python main.py --config configs/retinal_seed42.json
python main.py --config configs/bt_mri_seed42.json
```

Configurations are provided for seeds `42`, `123`, and `456`. During epochs 1-50, BiFormer uses learning rate `1e-4`; the CNN branches, GAM, and MLP use `1e-5`. From epoch 51, all modules use `1e-4` with cosine annealing (`T_max=250`, `eta_min=1e-6`). The checkpoint with highest validation accuracy is used for final test evaluation.

```text
outputs/<dataset>/<run_name>/
|-- checkpoints/
|   |-- best_accuracy.pt
|   |-- best_loss.pt
|   `-- last.pt
`-- results/
```

Training resumes from `last.pt` by default. Add `--no-resume` to start a new run.

## Evaluation

```bash
python evaluate.py \
  --config configs/retinal_seed42.json \
  --checkpoint /path/to/best_accuracy.pt \
  --data-root datasets/retinal
```

Reported metrics are Accuracy, Precision, Recall, F1-score, Sensitivity, Specificity, MCC, and AUC. Precision, Recall, F1-score, Sensitivity, and Specificity are support-weighted; AUC is micro-average ROC-AUC.

Evaluation saves JSON/CSV metrics, class-wise results, a confusion matrix, ROC curve, and precision-recall curve.

## Explainability

```bash
python explain.py \
  --bt-mri-config configs/bt_mri_seed42.json \
  --bt-mri-checkpoint /path/to/bt_mri_best_accuracy.pt \
  --bt-mri-data-root datasets/bt_mri \
  --retinal-config configs/retinal_seed42.json \
  --retinal-checkpoint /path/to/retinal_best_accuracy.pt \
  --retinal-data-root datasets/retinal \
  --output-dir results/xai
```

- Grad-CAM and Grad-CAM++ use `model.fusion.output_projection` (`[B, 768, 7, 7]`).
- t-SNE uses the 768-dimensional feature vector after global average pooling and before the classifier.
- Four random test images per class are selected with seed 2026.
- t-SNE features are standardized; the script uses seed 2026, perplexity up to 30, and 1,000 iterations.

Outputs are `gradcam_gradcamplusplus_vertical.png`, `tsne_bt_mri_gap.png`, and `tsne_retinal_gap.png`.

## Results

All checkpoints below were selected using validation accuracy. Values are test-set results.

### Retinal Eye Diseases

| Seed | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Sensitivity (%) | Specificity (%) | MCC (%) | AUC (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 95.14 | 95.13 | 95.14 | 95.13 | 95.14 | 98.40 | 93.52 | 99.47 |
| 123 | 94.36 | 94.38 | 94.36 | 94.34 | 94.36 | 98.15 | 92.49 | 99.31 |
| 456 | 93.57 | 93.66 | 93.57 | 93.55 | 93.57 | 97.92 | 91.47 | 99.50 |
| **Mean ± SD** | **94.36 ± 0.79** | **94.39 ± 0.74** | **94.36 ± 0.79** | **94.34 ± 0.79** | **94.36 ± 0.79** | **98.16 ± 0.24** | **92.49 ± 1.03** | **99.43 ± 0.10** |

### Brain Tumor MRI

| Seed | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Sensitivity (%) | Specificity (%) | MCC (%) | AUC (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 99.54 | 99.55 | 99.54 | 99.54 | 99.54 | 99.86 | 99.39 | 99.99 |
| 123 | 99.47 | 99.47 | 99.47 | 99.47 | 99.47 | 99.83 | 99.28 | 99.99 |
| 456 | 99.39 | 99.40 | 99.39 | 99.39 | 99.39 | 99.82 | 99.18 | 99.99 |
| **Mean ± SD** | **99.47 ± 0.08** | **99.47 ± 0.08** | **99.47 ± 0.08** | **99.47 ± 0.08** | **99.47 ± 0.08** | **99.84 ± 0.02** | **99.28 ± 0.11** | **99.99 ± 0.01** |

## Reproducibility

Seeds `42`, `123`, and `456` are provided. The code fixes Python, NumPy, PyTorch, and DataLoader seeds and stores RNG state in checkpoints. Exact values can still vary slightly with changes in GPU, CUDA, cuDNN, PyTorch, or driver version.

## Third-party code

BiFormer source code is included under [third_party/BiFormer/](third_party/BiFormer/). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
