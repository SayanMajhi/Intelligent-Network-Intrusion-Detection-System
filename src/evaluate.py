"""Metrics and plotting helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_curve)

sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#2A3F5F"]


def compute_metrics(y_true, y_pred, average="binary"):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if average == "binary":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m["detection_rate"] = tp / (tp + fn) if (tp + fn) else 0.0
        m["false_alarm_rate"] = fp / (fp + tn) if (fp + tn) else 0.0
    return m


def plot_confusion(y_true, y_pred, labels, title, path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(data, annot=cm if not normalize else np.round(data, 3), fmt=".3f" if normalize else "d",
                cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return cm


def plot_roc(curves, path, title="ROC curves on KDDTest+ (binary)"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, (name, y_true, score) in enumerate(curves):
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, lw=2 if "AESE" in name else 1.3, color=PALETTE[i % len(PALETTE)],
                label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_bar(df, metric_cols, path, title, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = df[metric_cols] * 100
    df_plot.plot(kind="bar", ax=ax, color=PALETTE[: len(metric_cols)], width=0.8)
    ax.set_ylabel("Score (%)"); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(loc="lower right", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=6, padding=1)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
