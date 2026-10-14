# XHybridFusion-Net: Attention-Guided BiFormer and Dual-CNN Learning for Explainable Medical
Image Classification

PyTorch implementation for four-class retinal disease and brain-tumor MRI
classification.

## Model

```text
Input -> BiFormer (4 blocks) -> DenseNet121 + EfficientNet-B0
      -> GAM Fusion -> Global Average Pooling -> GELU MLP -> Prediction
```

Default input size is `224 x 224`. GAM produces 49 tokens with 768 features.

Main configuration:

- BiFormer: 4 blocks, patch size 4, `n_win=7`, `topk=4`.
- CNN backbones: ImageNet-pretrained DenseNet121 and EfficientNet-B0.
- Fusion: channel attention, spatial attention, and 1x1 output projection.
- Classifier: `768 -> 1024 -> 512 -> 4` with GELU.
- Dropout and drop path: 0.1.
- Training: 300 epochs, batch size 32, AdamW, and cosine annealing.

## Project structure

```text
.
|-- train.py
|-- evaluate.py
|-- explain.py
|-- visualize_intermediate_feature_maps.py
|-- split_data.py
|-- configs/
|-- data/
|-- model/
|-- utils/
|-- results/
`-- third_party/BiFormer/
```

## Installation

```bash
python -m pip install -r requirements.txt
```

## Data

### Retinal

- Source: [Eye Diseases Classification on Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- Classes: cataract, diabetic retinopathy, glaucoma, normal
- Split used here: 70% train, 15% validation, 15% test
- Related source/patient images identified by the filename convention remain
  in the same split.

### BT-MRI

- Source: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1)
- Classes: glioma, meningioma, no tumor, pituitary
- The official Testing folder remains unchanged.
- The original Training folder is divided into 80% train and 20% validation.

Expected structure:

```text
datasets/<dataset>/
|-- train/<class>/
|-- val/<class>/
`-- test/<class>/
```

Use `split_data.py` to prepare the datasets.

```bash
python split_data.py \
  --dataset retinal \
  --source /path/to/raw/retinal \
  --output datasets/retinal \
  --seed 4242

python split_data.py \
  --dataset bt_mri \
  --source /path/to/raw/bt_mri \
  --output datasets/bt_mri \
  --seed 142
```

Training augmentation includes resize, color jitter, horizontal/vertical
flips, optional ±90-degree rotation, and RGB scaling to `[0, 1]`.

## Training

```bash
python train.py --config configs/retinal_seed42.json
python train.py --config configs/bt_mri_seed42.json
```

Configs for seeds `42`, `123`, and `456` are provided. Training saves
`best_accuracy.pt`, `best_loss.pt`, and `last.pt` under `outputs/`.

Training uses AdamW and a two-stage learning-rate schedule. During the first
50 epochs, BiFormer uses `1e-4` while both pretrained CNNs, GAM, and the MLP
use `1e-5`. All modules then use `1e-4` with cosine annealing to `1e-6`.

```text
outputs/<dataset>/<run_name>/
|-- checkpoints/
|   |-- best_accuracy.pt
|   |-- best_loss.pt
|   `-- last.pt
`-- results/
```

Training resumes automatically from `last.pt`. Use `--no-resume` to disable
resume behavior.

## Evaluation

```bash
python evaluate.py \
  --config configs/retinal_seed42.json \
  --checkpoint /path/to/best_accuracy.pt \
  --data-root /path/to/retinal
```

Reported metrics include Accuracy, Precision, Recall, F1-score, Sensitivity,
Specificity, MCC, and AUC.

Evaluation also saves the full JSON/CSV metrics, confusion matrix, ROC curve,
and precision-recall curve.

## Explainability

```bash
python explain.py \
  --bt-mri-config configs/bt_mri_seed42.json \
  --bt-mri-checkpoint /path/to/bt_mri_checkpoint.pt \
  --bt-mri-data-root /path/to/bt_mri \
  --retinal-config configs/retinal_seed42.json \
  --retinal-checkpoint /path/to/retinal_checkpoint.pt \
  --retinal-data-root /path/to/retinal
```

The scripts generate Grad-CAM, Grad-CAM++, t-SNE, and intermediate feature-map
figures.

For intermediate feature maps:

```bash
python visualize_intermediate_feature_maps.py --help
```

## Main results

### Retinal (300 Epochs)

| Seed | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Sensitivity (%) | Specificity (%) | MCC (%) | AUC (%) |
|:----:|-------------:|--------------:|-----------:|-------------:|----------------:|----------------:|--------:|--------:|
| 42 | 95.14 | 95.13 | 95.14 | 95.13 | 95.14 | 98.40 | 93.52 | 99.47 |
| 123 | 94.36 | 94.38 | 94.36 | 94.34 | 94.36 | 98.15 | 92.49 | 99.31 |
| 456 | 93.57 | 93.66 | 93.57 | 93.55 | 93.57 | 97.92 | 91.47 | 99.50 |
| **Mean ± std** | **94.36 ± 0.79** | **94.39 ± 0.74** | **94.36 ± 0.79** | **94.34 ± 0.79** | **94.36 ± 0.79** | **98.16 ± 0.24** | **92.49 ± 1.03** | **99.43 ± 0.10** |

### BT-MRI (300 Epochs)

| Seed | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Sensitivity (%) | Specificity (%) | MCC (%) | AUC (%) |
|:----:|-------------:|--------------:|-----------:|-------------:|----------------:|----------------:|--------:|--------:|
| 42 | 99.54 | 99.55 | 99.54 | 99.54 | 99.54 | 99.86 | 99.39 | 99.99 |
| 123 | 99.47 | 99.47 | 99.47 | 99.47 | 99.47 | 99.83 | 99.28 | 99.99 |
| 456 | 99.39 | 99.40 | 99.39 | 99.39 | 99.39 | 99.82 | 99.18 | 99.99 |
| **Mean ± std** | **99.47 ± 0.08** | **99.47 ± 0.08** | **99.47 ± 0.08** | **99.47 ± 0.08** | **99.47 ± 0.08** | **99.84 ± 0.02** | **99.28 ± 0.11** | **99.99 ± 0.01** |

Full metrics, classwise reports, ROC/PR curves, confusion matrices, and XAI
figures are available in [`results/`](results/).

## Reproducibility

The supplied configs use seeds 42, 123, and 456. DataLoader workers, Python,
NumPy, PyTorch, cuDNN determinism, TF32, and checkpoint RNG states are handled
by the project. Exact values may still vary across CUDA, cuDNN, PyTorch,
driver, and GPU versions.

## Third-party code

BiFormer source is included under `third_party/BiFormer/`. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.
