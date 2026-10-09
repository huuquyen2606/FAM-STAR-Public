"""FedProx baseline converted from the SOICT2026 10/20/50-client notebooks."""

from __future__ import annotations

import os
import time
from pathlib import Path

# Set this before importing PyTorch for deterministic CUDA RNNs.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.baselines.common.data import (
    HybridBLSTM_GRU,
    fed_avg,
    get_params,
    load_baseline_data,
    set_params,
)
from src.baselines.common.metrics import evaluate
from src.baselines.common.results import (
    build_payload_event,
    resolve_output_dir,
    save_model_checkpoint,
    save_results,
    summarize_payload,
)

# Settings are edited here; the module runner accepts no command-line flags.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_NAME = "FedProx"
METHOD_SLUG = "fedprox"
NUM_CLIENTS = 10
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

# Evaluated setting approved for K=10,20,50 (notebook source used 0.1).
FEDPROX_MU = 0.01


def local_train(
    model,
    loader,
    epochs: int,
    learning_rate: float,
    device,
    global_params,
    proximal_mu: float,
):
    model.train()
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)

            proximal_term = 0.0
            for name, param in model.named_parameters():
                global_param = global_params[name].to(device).detach()
                proximal_term += torch.sum((param - global_param) ** 2)
            loss = criterion(logits, y_batch) + (proximal_mu / 2.0) * proximal_term
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item())
            num_batches += 1

    return get_params(model), running_loss / max(num_batches, 1)


def train_federated(
    client_trainloaders,
    client_y_list,
    num_clients: int,
    global_testloader,
    num_features: int,
    num_classes: int,
    device,
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir: Path,
):
    device = torch.device(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_model = HybridBLSTM_GRU(
        input_size=num_features,
        num_classes=num_classes,
        blstm_hidden=BLSTM_HIDDEN,
        gru_hidden=GRU_HIDDEN,
        dense_hidden=DENSE_HIDDEN,
        dropout=DROPOUT,
    ).to(device)
    metrics_log = []
    payload_events = []

    print(
        f"Starting {METHOD_NAME}: {num_rounds} rounds | "
        f"{num_clients} clients | batch={batch_size} | mu={FEDPROX_MU}"
    )

    for round_id in range(1, num_rounds + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        global_params = get_params(global_model)
        download_payload = summarize_payload(global_params)
        local_t0 = time.perf_counter()
        client_params_list = []
        client_sizes = []
        client_train_losses = []

        for client_id in range(num_clients):
            local_model = HybridBLSTM_GRU(
                input_size=num_features,
                num_classes=num_classes,
                blstm_hidden=BLSTM_HIDDEN,
                gru_hidden=GRU_HIDDEN,
                dense_hidden=DENSE_HIDDEN,
                dropout=DROPOUT,
            ).to(device)
            set_params(local_model, global_params)
            payload_events.append(
                build_payload_event(
                    round_id, client_id, "downlink", "model", download_payload
                )
            )
            params, client_train_loss = local_train(
                model=local_model,
                loader=client_trainloaders[client_id],
                epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
                global_params=global_params,
                proximal_mu=FEDPROX_MU,
            )
            payload_events.append(
                build_payload_event(
                    round_id, client_id, "uplink", "update", summarize_payload(params)
                )
            )
            client_params_list.append(params)
            client_sizes.append(len(client_y_list[client_id]))
            client_train_losses.append(client_train_loss)

        local_time_sec = time.perf_counter() - local_t0
        server_t0 = time.perf_counter()
        set_params(global_model, fed_avg(client_params_list, client_sizes))
        metrics = evaluate(global_model, global_testloader, device, num_classes)
        server_time_sec = time.perf_counter() - server_t0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        metrics_log.append(
            {
                "round": round_id,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_macro": metrics["recall_macro"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "precision_micro": metrics["precision_micro"],
                "recall_micro": metrics["recall_micro"],
                "f1_micro": metrics["f1_micro"],
                "worst_class_f1": metrics["worst_class_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "train_loss": float(np.mean(client_train_losses)),
                "local_time_sec": round(local_time_sec, 2),
                "server_time_sec": round(server_time_sec, 2),
                "peak_vram_mb": round(float(peak_vram_mb), 2),
            }
        )

        # Unlike FedAvg, the source FedProx notebooks retain every round.
        save_model_checkpoint(
            global_model.state_dict(),
            checkpoint_dir / f"round_{round_id:02d}_fedprox.pt",
        )
        print(
            f"Round {round_id:02d}/{num_rounds} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1_W={metrics['f1_weighted']:.4f} | "
            f"F1_M={metrics['f1_macro']:.4f} | "
            f"Local={local_time_sec:.1f}s | Server={server_time_sec:.1f}s"
        )

    return {
        "model": global_model,
        "metrics_log": metrics_log,
        "payload_events": payload_events,
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
        num_features=data["num_features"],
        num_classes=data["num_classes"],
        device=data["device"],
        num_rounds=NUM_ROUNDS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=output_dir / "checkpoints",
    )
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
