"""Evaluation metrics shared by federated baselines."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    labels = list(range(num_classes))
    per_class_f1 = f1_score(
        all_labels,
        all_preds,
        average=None,
        labels=labels,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(
            all_labels, all_preds, labels=labels, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            all_labels, all_preds, labels=labels, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(
            all_labels, all_preds, labels=labels, average="macro", zero_division=0
        ),
        "recall_weighted": recall_score(
            all_labels, all_preds, labels=labels, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(
            all_labels, all_preds, labels=labels, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            all_labels, all_preds, labels=labels, average="weighted", zero_division=0
        ),
        "precision_micro": precision_score(
            all_labels, all_preds, labels=labels, average="micro", zero_division=0
        ),
        "recall_micro": recall_score(
            all_labels, all_preds, labels=labels, average="micro", zero_division=0
        ),
        "f1_micro": f1_score(
            all_labels, all_preds, labels=labels, average="micro", zero_division=0
        ),
        "worst_class_f1": float(per_class_f1.min()) if per_class_f1.size else 0.0,
        "preds": all_preds,
        "labels": all_labels,
    }
