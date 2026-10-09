import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def calculate_metrics(labels, predictions, probabilities=None, class_names=None):
    """Calculate multiclass metrics using one-vs-rest definitions.

    The primary precision, recall, F1, sensitivity, and specificity values use
    weighted averaging. Macro averages are also returned for comparison.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        number_of_classes = probabilities.shape[1]
    elif class_names is not None:
        number_of_classes = len(class_names)
    else:
        number_of_classes = int(max(labels.max(), predictions.max())) + 1

    class_ids = np.arange(number_of_classes)
    if class_names is None:
        class_names = [str(class_id) for class_id in class_ids]

    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    total = matrix.sum()

    true_positive = np.diag(matrix).astype(float)
    false_negative = matrix.sum(axis=1) - true_positive
    false_positive = matrix.sum(axis=0) - true_positive
    true_negative = total - true_positive - false_negative - false_positive

    sensitivity_per_class = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_negative) != 0,
    )
    specificity_per_class = np.divide(
        true_negative,
        true_negative + false_positive,
        out=np.zeros_like(true_negative),
        where=(true_negative + false_positive) != 0,
    )

    support = matrix.sum(axis=1).astype(float)
    support_total = support.sum()
    if support_total == 0:
        weighted_sensitivity = 0.0
        weighted_specificity = 0.0
    else:
        weighted_sensitivity = float(
            np.sum(sensitivity_per_class * support) / support_total
        )
        weighted_specificity = float(
            np.sum(specificity_per_class * support) / support_total
        )

    class_precision, class_recall, class_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=class_ids,
            average=None,
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(
            labels, predictions, average="weighted", zero_division=0
        )),
        "recall": float(recall_score(
            labels, predictions, average="weighted", zero_division=0
        )),
        "f1_score": float(f1_score(
            labels, predictions, average="weighted", zero_division=0
        )),
        # In multiclass one-vs-rest evaluation, sensitivity equals recall.
        "sensitivity": weighted_sensitivity,
        "specificity": weighted_specificity,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "precision_macro": float(precision_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "f1_score_macro": float(f1_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "sensitivity_macro": float(np.mean(sensitivity_per_class)),
        "specificity_macro": float(np.mean(specificity_per_class)),
        "confusion_matrix": matrix.tolist(),
    }

    binary_labels = label_binarize(labels, classes=class_ids)
    auc_per_class = [None] * number_of_classes
    if probabilities is not None:
        try:
            # The primary AUC is the micro-average ROC AUC.
            metrics["auc"] = float(roc_auc_score(
                binary_labels.ravel(), probabilities.ravel()
            ))
        except ValueError:
            metrics["auc"] = None

        try:
            metrics["auc_macro"] = float(roc_auc_score(
                labels,
                probabilities,
                labels=class_ids,
                average="macro",
                multi_class="ovr",
            ))
        except ValueError:
            metrics["auc_macro"] = None

        for class_id in class_ids:
            try:
                auc_per_class[class_id] = float(roc_auc_score(
                    binary_labels[:, class_id], probabilities[:, class_id]
                ))
            except ValueError:
                auc_per_class[class_id] = None
    else:
        metrics["auc"] = None
        metrics["auc_macro"] = None

    classwise = {}
    for class_id, class_name in enumerate(class_names):
        classwise[class_name] = {
            "precision": float(class_precision[class_id]),
            "recall": float(class_recall[class_id]),
            "f1_score": float(class_f1[class_id]),
            "sensitivity": float(sensitivity_per_class[class_id]),
            "specificity": float(specificity_per_class[class_id]),
            "auc": auc_per_class[class_id],
            "support": int(support[class_id]),
        }

    metrics["classwise"] = classwise
    return metrics
