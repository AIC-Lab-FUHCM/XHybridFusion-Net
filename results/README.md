# Results

This directory contains the curated outputs reported with the project.

## Evaluation

`evaluation_cosine300/` contains results from the best-validation-accuracy
checkpoints:

- `metrics_all_checkpoints.csv`: combined percentage metrics.
- `metrics_table.png`: summary table.
- `<dataset>/Seed42/metrics.json`: full metrics, classwise scores, checkpoint
  metadata, and confusion matrix values for the selected seed-42 run.
- `<dataset>/Seed42/confusion_matrix.png`: confusion matrix.
- `<dataset>/Seed42/roc_curve.png`: classwise and micro-average ROC curves.
- `<dataset>/Seed42/precision_recall_curve.png`: classwise and micro-average
  precision-recall curves.

The primary precision, recall, F1-score, sensitivity, and specificity values
use weighted averaging. The primary AUC is the micro-average one-vs-rest ROC
AUC. All values in the combined CSV are percentages.

## Explainability

- `xai_bt_mri_gradcam_retinal_tsne/` contains Grad-CAM, Grad-CAM++, and t-SNE
  outputs generated from the seed-42 checkpoints.
- `intermediate_feature_maps/` contains representative intermediate feature
  maps for BT-MRI and retinal images.

Model checkpoints and datasets are intentionally not included in the
repository.
