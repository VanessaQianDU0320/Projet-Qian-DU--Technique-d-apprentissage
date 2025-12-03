# %%
import numpy as np
import os

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics["f1"]        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"]  = average_precision_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"]  = None

    return metrics

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix

FIG_DIR = "reports/figures"
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)


def plot_roc_pr_curves(y_true, y_proba, model_name, scaler_name):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.3f})")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve - {model_name} ({scaler_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ROC_{model_name}_{scaler_name}.png", dpi=180)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AUC={auc(recall, precision):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {model_name} ({scaler_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/PR_{model_name}_{scaler_name}.png", dpi=180)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, scaler_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name} ({scaler_name})")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/CM_{model_name}_{scaler_name}.png", dpi=180)
    plt.close()
