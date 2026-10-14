# Pretrained checkpoints

The `.pt`, `.pth`, and `.ckpt` model-weight files are intentionally excluded from this repository because of their size.

## Download

All checkpoints are available in one Google Drive folder:

**Google Drive folder:** [Download all checkpoints](https://drive.google.com/drive/folders/1aWevxTZeP6_TU2Bs0NP-CGDBnEbsgqLQ)

Download the checkpoint needed for the selected dataset and seed before
evaluation or XAI.

Download a checkpoint and provide its local path through the `--checkpoint` option in `evaluate.py`, or through the relevant `--*-checkpoint` option in `explain.py`.

Example:

```bash
python evaluate.py \
  --config configs/bt_mri_seed42.json \
  --checkpoint checkpoints/bt_mri_seed42_best_accuracy.pt \
  --data-root datasets/bt_mri
```
