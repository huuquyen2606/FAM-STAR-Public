"""FedGKD-VOTE port of the Distillation/FedGKD-FedGKDVOTE/FedGKD_VOTE notebooks.

The 10-client notebook accounts for older-teacher downlinks; the 20/50-client
notebooks account only for the current global model. Preserve those source
variants without inventing caching. Validation selects vote weights and the
best checkpoint; final reporting still evaluates the last-round model.
"""

from __future__ import annotations

import copy
import os
import time
from collections import OrderedDict
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.baselines.common.data import (
    HybridBLSTM_GRU,
    get_params,
    load_baseline_data,
    set_params,
    set_seed,
)
from src.baselines.common.metrics import evaluate
from src.baselines.common.results import (
    build_payload_event,
    resolve_output_dir,
    save_model_checkpoint,
    save_results,
    summarize_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_NAME = "FedGKD-VOTE"
METHOD_SLUG = "fedgkd_vote"
NUM_CLIENTS = 10  # Editable to 20 or 50, with the corresponding prepared dataset.
NUM_ROUNDS = 50
NUM_EPOCHS = 5
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
LEARNING_RATE = 0.002
SEED = 42
BLSTM_HIDDEN = 300
GRU_HIDDEN = 100
DENSE_HIDDEN = 80
DROPOUT = 0.3
DATA_DIR = None
OUTPUT_DIR = None
DEVICE = None
FEDGKD_M = 5
FEDGKD_GAMMA = 0.5
TEMPERATURE = 2.0
FEDGKD_BETA = 1.0 / FEDGKD_M
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = None
# Preserve the notebook-specific accounting, not a new transport policy.
HISTORICAL_TEACHER_ACCOUNTING_CLIENT_COUNTS = {10}


def build_model(num_features, num_classes, device):
    return HybridBLSTM_GRU(
        input_size=num_features,
        num_classes=num_classes,
        blstm_hidden=BLSTM_HIDDEN,
        gru_hidden=GRU_HIDDEN,
        dense_hidden=DENSE_HIDDEN,
        dropout=DROPOUT,
        flatten_parameters=True,
    ).to(device)


def fedavg_weighted(client_params_list, client_sizes):
    """Retain VOTE's float32 multiply-then-add order, not add_(alpha=...)."""
    total = float(sum(client_sizes))
    avg_params = OrderedDict()
    for key in client_params_list[0].keys():
        first_tensor = client_params_list[0][key]
        if not torch.is_floating_point(first_tensor):
            avg_params[key] = first_tensor.clone()
            continue
        avg_tensor = torch.zeros_like(first_tensor, dtype=torch.float32)
        for params, size in zip(client_params_list, client_sizes):
            avg_tensor += params[key].detach().float() * (float(size) / total)
        avg_params[key] = avg_tensor.to(dtype=first_tensor.dtype)
    return avg_params


def calculate_vote_weights(server_buffer, beta):
    """Softmax(-validation_loss/beta), in the notebook's NumPy float64 order."""
    losses = np.array([item["val_loss"] for item in server_buffer], dtype=np.float64)
    scaled = -losses / beta
    shifted = scaled - np.max(scaled)
    exp_weights = np.exp(shifted)
    weights = exp_weights / np.sum(exp_weights)
    return weights.tolist()


def train_one_client_vote(
    model,
    loader,
    *,
    epochs,
    lr,
    device,
    teachers_list=None,
    teacher_weights=None,
):
    """CE plus the weighted sum of individual temperature-squared teacher KLs."""
    model.train()
    if teachers_list is not None:
        for teacher in teachers_list:
            teacher.eval()
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_ce, total_kd = 0.0, 0.0, 0.0
    total_correct, total_seen = 0, 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            ce_loss = criterion(logits, yb)
            kd_loss = torch.tensor(0.0, device=device)
            if teachers_list is not None and teacher_weights is not None:
                student_log_probs = F.log_softmax(logits / TEMPERATURE, dim=1)
                for teacher, weight in zip(teachers_list, teacher_weights):
                    with torch.no_grad():
                        teacher_logits = teacher(xb)
                    kl = F.kl_div(
                        student_log_probs,
                        F.softmax(teacher_logits / TEMPERATURE, dim=1),
                        reduction="batchmean",
                    ) * (TEMPERATURE**2)
                    kd_loss = kd_loss + float(weight) * kl
                loss = ce_loss + FEDGKD_GAMMA * kd_loss
            else:
                loss = ce_loss
            loss.backward()
            if GRAD_CLIP_NORM:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_correct += (preds == yb).sum().item()
                total_seen += yb.size(0)
                total_loss += loss.item() * yb.size(0)
                total_ce += ce_loss.item() * yb.size(0)
                total_kd += kd_loss.item() * yb.size(0)
    n_samples = max(total_seen, 1)
    return get_params(model), {
        "loss": total_loss / n_samples,
        "ce_loss": total_ce / n_samples,
        "kd_loss": total_kd / n_samples,
        "acc": total_correct / n_samples,
        "samples": len(loader.dataset),
    }


def train_federated(
    *,
    client_trainloaders,
    client_y_list,
    num_clients,
    global_testloader,
    global_valloader,
    num_features,
    num_classes,
    device,
    num_rounds,
    num_epochs,
    batch_size,
    learning_rate,
    checkpoint_dir,
):
    # The source re-seeds immediately before its federated loop, after its
    # display-only architecture probe. Removing that probe does not change RNG.
    set_seed(SEED)
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_model = build_model(num_features, num_classes, device)
    server_buffer = []
    metrics_log, payload_events = [], []
    best_score, best_round, best_params = -1.0, None, None
    teacher_weights = None
    local_losses, local_ce_losses, local_kd_losses, local_accs = [], [], [], []
    eval_res, val_res = None, None
    print(
        f"Starting {METHOD_NAME}: M={FEDGKD_M} | {num_rounds} rounds | "
        f"{num_clients} clients | batch={batch_size}"
    )

    for rnd in range(1, num_rounds + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        global_params = get_params(global_model)
        download_payload = summarize_payload(global_params)
        teachers_list, teacher_weights = None, None
        if len(server_buffer) > 0:
            teacher_weights = calculate_vote_weights(server_buffer, beta=FEDGKD_BETA)
            teachers_list = []
            for item in server_buffer:
                teacher = build_model(num_features, num_classes, device)
                set_params(teacher, item["params"])
                teacher.eval()
                for param in teacher.parameters():
                    param.requires_grad = False
                teachers_list.append(teacher)

        client_params_list, client_sizes = [], []
        local_losses, local_ce_losses, local_kd_losses, local_accs = [], [], [], []
        local_t0 = time.perf_counter()
        for cid in range(num_clients):
            local_model = build_model(num_features, num_classes, device)
            set_params(local_model, global_params)
            # The current global model is also the buffer's newest teacher.
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "downlink",
                    "model",
                    download_payload,
                )
            )
            params, info = train_one_client_vote(
                model=local_model,
                loader=client_trainloaders[cid],
                epochs=num_epochs,
                lr=learning_rate,
                device=device,
                teachers_list=teachers_list,
                teacher_weights=teacher_weights,
            )
            client_params_list.append(params)
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "uplink",
                    "update",
                    summarize_payload(params),
                )
            )
            client_sizes.append(info["samples"])
            local_losses.append(info["loss"])
            local_ce_losses.append(info["ce_loss"])
            local_kd_losses.append(info["kd_loss"])
            local_accs.append(info["acc"])
            del local_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        local_time_sec = time.perf_counter() - local_t0

        # Only the 10-client source logs these extra downlinks, outside local timing.
        if (
            num_clients in HISTORICAL_TEACHER_ACCOUNTING_CLIENT_COUNTS
            and len(server_buffer) > 1
        ):
            for cid in range(num_clients):
                for teacher_idx in range(len(server_buffer) - 1):
                    payload_events.append(
                        build_payload_event(
                            rnd,
                            cid,
                            "downlink",
                            f"vote_teacher_{teacher_idx}",
                            download_payload,
                        )
                    )
        if teachers_list is not None:
            del teachers_list
            if device.type == "cuda":
                torch.cuda.empty_cache()

        server_t0 = time.perf_counter()
        averaged_params = fedavg_weighted(client_params_list, client_sizes)
        set_params(global_model, averaged_params)
        eval_res = evaluate(
            global_model,
            global_testloader,
            device,
            num_classes,
            include_loss=True,
            include_mcc=True,
        )
        val_res = evaluate(
            global_model,
            global_valloader,
            device,
            num_classes,
            include_loss=True,
            include_mcc=True,
        )
        if len(server_buffer) >= FEDGKD_M:
            server_buffer.pop(0)
        server_buffer.append(
            {
                "params": copy.deepcopy(averaged_params),
                "val_loss": val_res["loss"],
            }
        )
        server_time_sec = time.perf_counter() - server_t0
        train_loss = float(np.mean(local_losses))
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        metrics_log.append(
            {
                "round": rnd,
                **{
                    key: value
                    for key, value in eval_res.items()
                    if key not in ("preds", "labels")
                },
                "train_loss": train_loss,
                "local_time_sec": round(local_time_sec, 2),
                "server_time_sec": round(server_time_sec, 2),
                "peak_vram_mb": round(float(peak_vram_mb), 2),
                "val_loss": val_res["loss"],
                "val_f1_macro": val_res["f1_macro"],
                "avg_local_loss": float(np.mean(local_losses)),
                "avg_local_ce": float(np.mean(local_ce_losses)),
                "avg_local_kd": float(np.mean(local_kd_losses)),
                "avg_local_acc": float(np.mean(local_accs)),
                "buffer_size": len(server_buffer),
                "kd_enabled": rnd > 1,
            }
        )
        if val_res["f1_macro"] > best_score:
            best_score = val_res["f1_macro"]
            best_round = rnd
            best_params = copy.deepcopy(averaged_params)
        ckpt_path = checkpoint_dir / f"round_{rnd:02d}_fedgkd_vote.pt"
        save_model_checkpoint(averaged_params, ckpt_path)
        print(
            f"Round {rnd:02d}/{num_rounds:02d} | KD={'OFF' if rnd == 1 else 'ON '} "
            f"(Buf={len(server_buffer)}/{FEDGKD_M}) | Loss={eval_res['loss']:.4f} | "
            f"Acc={eval_res['accuracy']:.4f} | F1_M={eval_res['f1_macro']:.4f} | "
            f"F1_W={eval_res['f1_weighted']:.4f} | KD_Loss={np.mean(local_kd_losses):.4f} | "
            f"Local={local_time_sec:.1f}s | Server={server_time_sec:.1f}s"
        )

    best_ckpt_path = checkpoint_dir / "best_model_fedgkd_vote.pt"
    save_model_checkpoint(best_params, best_ckpt_path)
    print(
        f"Best Round: {best_round} | Best Validation F1-Macro: {best_score:.4f} "
        f"(Saved: {best_ckpt_path})"
    )
    results_df = pd.DataFrame(metrics_log)
    best_r = results_df.loc[results_df["val_f1_macro"].idxmax()]
    last_r = results_df.iloc[-1]
    summary_df = pd.DataFrame(
        [
            {
                "method": METHOD_NAME,
                "best_round": int(best_r["round"]),
                "best_accuracy": best_r["accuracy"],
                "best_f1_macro": best_r["f1_macro"],
                "best_f1_weighted": best_r["f1_weighted"],
                "last_round": int(last_r["round"]),
                "last_accuracy": last_r["accuracy"],
                "last_f1_macro": last_r["f1_macro"],
                "last_f1_weighted": last_r["f1_weighted"],
                "buffer_M": FEDGKD_M,
                "gamma": FEDGKD_GAMMA,
                "temperature": TEMPERATURE,
            }
        ]
    )
    return {
        "model": global_model,
        "metrics_log": metrics_log,
        "payload_events": payload_events,
        "history": server_buffer,
        "best_params": best_params,
        "best_score": best_score,
        "best_round": best_round,
        "best_checkpoint": best_ckpt_path,
        "teacher_weights": teacher_weights,
        "local_losses": local_losses,
        "local_ce_losses": local_ce_losses,
        "local_kd_losses": local_kd_losses,
        "local_accs": local_accs,
        "final_metrics": eval_res,
        "validation_metrics": val_res,
        "summary": summary_df,
    }


def main() -> int:
    data = load_baseline_data(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        seed=SEED,
        project_root=PROJECT_ROOT,
        data_dir=DATA_DIR,
        device=DEVICE,
        validation=True,
    )
    output_dir = (
        Path(OUTPUT_DIR)
        if OUTPUT_DIR is not None
        else resolve_output_dir(METHOD_SLUG, NUM_CLIENTS, PROJECT_ROOT)
    )
    result = train_federated(
        client_trainloaders=data["client_trainloaders"],
        client_y_list=data["client_y_list"],
        num_clients=NUM_CLIENTS,
        global_testloader=data["global_testloader"],
        global_valloader=data["global_valloader"],
        num_features=data["num_features"],
        num_classes=data["num_classes"],
        device=data["device"],
        num_rounds=NUM_ROUNDS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=output_dir / "checkpoints",
    )
    # Do not restore best_params: the source reports the final round endpoint.
    save_results(
        output_dir=output_dir,
        classes=data["classes"],
        num_classes=data["num_classes"],
        num_rounds=NUM_ROUNDS,
        metrics_log=result["metrics_log"],
        payload_events=result["payload_events"],
        global_model=result["model"],
        global_testloader=data["global_testloader"],
        device=data["device"],
        method_name=METHOD_NAME,
        include_classification_report=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
