"""Payload accounting and result export helpers for FL baselines."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .metrics import evaluate


def resolve_output_dir(
    method_name: str,
    num_clients: int,
    project_root: str | Path,
) -> Path:
    """Separate each method's results into its client-count folders."""
    client_folder = f"{num_clients}clients"
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        return kaggle_working / method_name / client_folder
    return Path(project_root) / "experiments" / method_name / client_folder


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
    "prototype_bytes",
    "metadata_bytes",
    "nnz",
    "total_params",
    "sparsity",
    "error_norm",
]
COMMUNICATION_COLUMNS = [
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


def summarize_payload(state_dict, return_wire: bool = False):
    """Measure the dense wire package; JoPEQ also consumes its actual bytes."""
    cpu_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }
    total_params = int(sum(tensor.numel() for tensor in cpu_state.values()))
    nnz = int(sum(torch.count_nonzero(tensor).item() for tensor in cpu_state.values()))
    values_bytes = int(
        sum(tensor.numel() * tensor.element_size() for tensor in cpu_state.values())
    )

    serialized_buffer = io.BytesIO()
    torch.save(cpu_state, serialized_buffer)
    serialized_bytes = int(serialized_buffer.getbuffer().nbytes)
    metadata_bytes = max(serialized_bytes - values_bytes, 0)
    sparsity = 1.0 - (nnz / max(total_params, 1))

    summary = {
        "raw_bytes": values_bytes,
        "serialized_bytes": serialized_bytes,
        "values_bytes": values_bytes,
        "indices_bytes": 0,
        "scales_bytes": 0,
        "mask_bytes": 0,
        "prototype_bytes": 0,
        "metadata_bytes": metadata_bytes,
        "nnz": nnz,
        "total_params": total_params,
        "sparsity": float(sparsity),
    }
    if return_wire:
        return serialized_buffer.getvalue(), summary
    return summary


def build_payload_event(
    round_id,
    client_id,
    direction,
    payload_type,
    summary,
    error_norm: float = 0.0,
    **extra,
):
    return {
        "round": round_id,
        "client_id": client_id,
        "direction": direction,
        "payload_type": payload_type,
        **summary,
        "error_norm": float(error_norm),
        **extra,
    }


def save_model_checkpoint(params, path: str | Path) -> None:
    """Save notebook model weights; retention and auxiliary state are method-owned."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in params.items()},
        path,
    )
    print(f"--> Checkpoint saved to {path}")


def save_results(
    output_dir: str | Path,
    classes: list[str],
    num_classes: int,
    num_rounds: int,
    metrics_log: list[dict],
    payload_events: list[dict],
    global_model,
    global_testloader,
    device,
    method_name: str,
    include_classification_report: bool = False,
    extra_tables: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(metrics_log)
    extra_metrics_columns = [
        column for column in results_df.columns if column not in METRICS_COLUMNS
    ]
    results_df = results_df.reindex(columns=METRICS_COLUMNS + extra_metrics_columns)
    results_df.to_csv(output_dir / "metrics_per_round.csv", index=False)

    payload_events_df = pd.DataFrame(payload_events)
    extra_payload_columns = [
        column for column in payload_events_df.columns if column not in PAYLOAD_COLUMNS
    ]
    payload_events_df = payload_events_df.reindex(
        columns=PAYLOAD_COLUMNS + extra_payload_columns
    )
    payload_events_df.to_csv(output_dir / "payload_events.csv", index=False)

    uplink_df = payload_events_df[payload_events_df["direction"] == "uplink"].copy()
    downlink_df = payload_events_df[payload_events_df["direction"] == "downlink"].copy()
    uplink_df["serialized_bytes"] = pd.to_numeric(
        uplink_df["serialized_bytes"], errors="coerce"
    ).fillna(0)
    downlink_df["serialized_bytes"] = pd.to_numeric(
        downlink_df["serialized_bytes"], errors="coerce"
    ).fillna(0)
    uplink_df["nnz"] = pd.to_numeric(uplink_df["nnz"], errors="coerce").fillna(0)
    uplink_df["total_params"] = pd.to_numeric(
        uplink_df["total_params"], errors="coerce"
    ).fillna(0)

    upload_by_round = uplink_df.groupby("round", as_index=False).agg(
        participants=("client_id", "nunique"),
        upload_bytes=("serialized_bytes", "sum"),
        avg_nnz=("nnz", "mean"),
        avg_total_params=("total_params", "mean"),
    )
    download_by_round = downlink_df.groupby("round", as_index=False).agg(
        download_bytes=("serialized_bytes", "sum")
    )
    communication_rounds_df = (
        upload_by_round.merge(download_by_round, on="round", how="outer")
        .fillna(0)
        .sort_values("round")
        .reset_index(drop=True)
    )
    communication_rounds_df["participants"] = communication_rounds_df[
        "participants"
    ].astype(int)
    communication_rounds_df["upload_bytes"] = (
        communication_rounds_df["upload_bytes"].round().astype("int64")
    )
    communication_rounds_df["download_bytes"] = (
        communication_rounds_df["download_bytes"].round().astype("int64")
    )
    communication_rounds_df["total_bytes"] = (
        communication_rounds_df["upload_bytes"]
        + communication_rounds_df["download_bytes"]
    )
    participants_safe = communication_rounds_df["participants"].replace(0, np.nan)
    communication_rounds_df["upload_MB_client_round"] = (
        communication_rounds_df["upload_bytes"] / participants_safe / 1e6
    ).fillna(0.0)
    communication_rounds_df["download_MB_client_round"] = (
        communication_rounds_df["download_bytes"] / participants_safe / 1e6
    ).fillna(0.0)
    communication_rounds_df["cumulative_total_GB"] = (
        communication_rounds_df["total_bytes"].cumsum() / 1e9
    )
    communication_rounds_df["effective_sparsity"] = (
        1.0
        - communication_rounds_df["avg_nnz"]
        / communication_rounds_df["avg_total_params"].replace(0, np.nan)
    ).fillna(0.0)
    communication_rounds_df = communication_rounds_df[COMMUNICATION_COLUMNS]
    communication_rounds_df.to_csv(
        output_dir / "communication_rounds.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    metric_series = [
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
    for column, label, marker in metric_series:
        ax.plot(results_df["round"], results_df[column], label=label, marker=marker)
    ax.set_title(f"Metrics Convergence across {num_rounds} Rounds ({method_name})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_per_round.png")
    plt.close(fig)

    final_metrics = evaluate(global_model, global_testloader, device, num_classes)
    y_pred = final_metrics["preds"]
    y_true = final_metrics["labels"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
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
        output_dir / "per_class_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - Final Round ({method_name})")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    if include_classification_report:
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
        (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        print(report)

    for filename, rows in (extra_tables or {}).items():
        table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        table_path = output_dir / filename
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)

    print("\nFinal round metrics:")
    print(results_df.tail(1).to_string(index=False))
    print(f"Saved results under: {output_dir}")
    return final_metrics
