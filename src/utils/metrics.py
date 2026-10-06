"""CSV and figure export for FAM-STAR experiment results."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


METRICS_COLUMNS = [
    "round",
    "accuracy",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "worst_class_f1",
    "balanced_accuracy",
    "train_loss",
    "local_time_sec",
    "server_time_sec",
    "peak_vram_mb",
]

PAYLOAD_COLUMNS = [
    "round",
    "client_id",
    "direction",
    "payload_type",
    "raw_bytes",
    "serialized_bytes",
    "values_bytes",
    "indices_bytes",
    "scales_bytes",
    "mask_bytes",
    "support_bitmap_bytes",
    "proposal_bitmap_bytes",
    "broadcast_mask_bitmap_bytes",
    "prototype_bytes",
    "metadata_bytes",
    "nnz",
    "total_params",
    "sparsity",
    "error_norm",
]


def export_results(
    metrics_log,
    payload_events,
    final_metrics,
    classes,
    output_dir: str,
    num_rounds: int,
):
    """Export the notebook's metrics/communication outputs into clean folders."""
    metrics_dir = os.path.join(output_dir, "metrics")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    metrics_df = pd.DataFrame(
        metrics_log
    ).reindex(columns=METRICS_COLUMNS)
    metrics_df.to_csv(
        os.path.join(metrics_dir, "metrics_per_round.csv"),
        index=False,
    )

    payload_df = (
        pd.DataFrame(payload_events)
        .reindex(columns=PAYLOAD_COLUMNS)
        .rename(
            columns={
                "prototype_bytes": "prototype_value_bytes"
            }
        )
    )

    wire_component_columns = [
        "values_bytes",
        "indices_bytes",
        "scales_bytes",
        "mask_bytes",
        "prototype_value_bytes",
        "metadata_bytes",
    ]
    bitmap_columns = [
        "support_bitmap_bytes",
        "proposal_bitmap_bytes",
        "broadcast_mask_bitmap_bytes",
    ]

    for column in [
        "serialized_bytes",
        *wire_component_columns,
        *bitmap_columns,
    ]:
        payload_df[column] = pd.to_numeric(
            payload_df[column],
            errors="coerce",
        ).fillna(0).round().astype("int64")

    accounted_bytes = payload_df[
        wire_component_columns
    ].sum(axis=1)
    if not accounted_bytes.equals(
        payload_df["serialized_bytes"]
    ):
        raise AssertionError(
            "Serialized communication bytes are not fully accounted for"
        )

    payload_df.to_csv(
        os.path.join(metrics_dir, "payload_events.csv"),
        index=False,
    )

    uplink_df = payload_df[
        payload_df["direction"] == "uplink"
    ].copy()
    downlink_df = payload_df[
        payload_df["direction"] == "downlink"
    ].copy()

    uplink_df["nnz"] = pd.to_numeric(
        uplink_df["nnz"],
        errors="coerce",
    ).fillna(0)
    uplink_df["total_params"] = pd.to_numeric(
        uplink_df["total_params"],
        errors="coerce",
    ).fillna(0)

    upload_by_round = uplink_df.groupby(
        "round",
        as_index=False,
    ).agg(
        participants=("client_id", "nunique"),
        upload_bytes=("serialized_bytes", "sum"),
        avg_nnz=("nnz", "mean"),
        avg_total_params=("total_params", "mean"),
    )
    download_by_round = downlink_df.groupby(
        "round",
        as_index=False,
    ).agg(
        download_bytes=("serialized_bytes", "sum"),
    )

    communication_df = upload_by_round.merge(
        download_by_round,
        on="round",
        how="outer",
    ).fillna(0)
    communication_df = (
        communication_df
        .sort_values("round")
        .reset_index(drop=True)
    )

    communication_df["participants"] = (
        communication_df["participants"].astype(int)
    )
    communication_df["upload_bytes"] = (
        communication_df["upload_bytes"]
        .round()
        .astype("int64")
    )
    communication_df["download_bytes"] = (
        communication_df["download_bytes"]
        .round()
        .astype("int64")
    )
    communication_df["total_bytes"] = (
        communication_df["upload_bytes"]
        + communication_df["download_bytes"]
    )

    safe_participants = communication_df[
        "participants"
    ].replace(0, np.nan)
    communication_df["upload_MB_client_round"] = (
        communication_df["upload_bytes"]
        / safe_participants
        / 1e6
    ).fillna(0.0)
    communication_df["download_MB_client_round"] = (
        communication_df["download_bytes"]
        / safe_participants
        / 1e6
    ).fillna(0.0)
    communication_df["cumulative_total_GB"] = (
        communication_df["total_bytes"].cumsum()
        / 1e9
    )
    communication_df["effective_sparsity"] = (
        1.0
        - communication_df["avg_nnz"]
        / communication_df["avg_total_params"].replace(
            0,
            np.nan,
        )
    ).fillna(0.0)

    communication_columns = [
        "round",
        "participants",
        "upload_bytes",
        "download_bytes",
        "total_bytes",
        "upload_MB_client_round",
        "download_MB_client_round",
        "cumulative_total_GB",
        "avg_nnz",
        "effective_sparsity",
    ]
    communication_df = communication_df[
        communication_columns
    ]
    communication_df.to_csv(
        os.path.join(
            metrics_dir,
            "communication_rounds.csv",
        ),
        index=False,
    )

    # Convergence figure.
    plt.figure(figsize=(12, 8))
    series = [
        ("accuracy", "Accuracy", "o"),
        ("precision_macro", "Precision (Macro)", "x"),
        ("precision_weighted", "Precision (Weighted)", "X"),
        ("recall_macro", "Recall (Macro)", "s"),
        ("recall_weighted", "Recall (Weighted)", "d"),
        ("f1_macro", "F1-Score (Macro)", "v"),
        ("f1_weighted", "F1-Score (Weighted)", "^"),
        ("precision_micro", "Precision (Micro)", "+"),
        ("recall_micro", "Recall (Micro)", "*"),
        ("f1_micro", "F1-Score (Micro)", "P"),
    ]
    for column, label, marker in series:
        plt.plot(
            metrics_df["round"],
            metrics_df[column],
            label=label,
            marker=marker,
        )
    plt.title(
        f"Metrics Convergence across {num_rounds} Rounds (FAM-STAR)"
    )
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figures_dir,
            "metrics_per_round.png",
        )
    )
    plt.close()

    y_pred = final_metrics["preds"]
    y_true = final_metrics["labels"]
    class_labels = list(range(len(classes)))

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_labels,
            zero_division=0,
        )
    )
    per_class_df = pd.DataFrame(
        {
            "class": classes,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )
    per_class_df.to_csv(
        os.path.join(
            metrics_dir,
            "per_class_metrics.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=classes,
        zero_division=0,
    )
    with open(
        os.path.join(
            metrics_dir,
            "classification_report.txt",
        ),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(report)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels,
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(
        "Confusion Matrix - Final Round (FAM-STAR)"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figures_dir,
            "confusion_matrix.png",
        )
    )
    plt.close()

    return {
        "metrics_per_round": metrics_df,
        "payload_events": payload_df,
        "communication_rounds": communication_df,
        "per_class_metrics": per_class_df,
    }
