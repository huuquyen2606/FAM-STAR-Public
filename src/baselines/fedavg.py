"""FedAvg 10-client runner ported from the original notebook."""

from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path

# PyTorch requires this before CUDA initializes for deterministic RNNs.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.baselines.common.data import (
    load_cicandmal_data,
    make_client_loaders,
    resolve_cicandmal_data_dir,
)
from src.baselines.common.metrics import evaluate
from src.baselines.common.results import (
    archive_results,
    resolve_output_dir,
    save_results,
    summarize_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLIENTS = 10
NUM_ROUNDS = 50
NUM_EPOCHS = 5
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
LEARNING_RATE = 0.002
SEED = 42

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class HybridBLSTM_GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        blstm_hidden: int = 300,
        gru_hidden: int = 100,
        dense_hidden: int = 80,
    ) -> None:
        super().__init__()

        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=blstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.ln_blstm = nn.LayerNorm(blstm_hidden * 2)
        self.gru = nn.GRU(
            input_size=blstm_hidden * 2,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.ln_gru = nn.LayerNorm(gru_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(gru_hidden, dense_hidden)
        self.ln_fc = nn.LayerNorm(dense_hidden)
        self.fc_out = nn.Linear(dense_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.blstm(x)
        out = out[:, -1, :]
        out = self.relu(self.ln_blstm(out))
        out = out.unsqueeze(1)
        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.ln_gru(out)))
        out = self.relu(self.ln_fc(self.fc1(out)))
        return self.fc_out(out)


def get_params(model: nn.Module):
    return copy.deepcopy(model.state_dict())


def set_params(model: nn.Module, params) -> None:
    model.load_state_dict(params)


def fed_avg(client_params_list, client_sizes):
    total = sum(client_sizes)
    avg = {}
    for key in client_params_list[0]:
        first = client_params_list[0][key]
        if torch.is_floating_point(first):
            avg[key] = torch.zeros_like(first)
            for params, size in zip(client_params_list, client_sizes):
                avg[key].add_(params[key].to(dtype=avg[key].dtype), alpha=size / total)
        else:
            avg[key] = first.clone()
    return avg


def local_train(model, loader, epochs: int, learning_rate: float, device):
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
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item())
            num_batches += 1

    mean_loss = running_loss / max(num_batches, 1)
    return get_params(model), mean_loss


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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_model = HybridBLSTM_GRU(
        input_size=num_features,
        num_classes=num_classes,
    ).to(device)
    metrics_log = []
    payload_events = []

    print(
        f"Starting Federated Learning: {num_rounds} rounds | "
        f"{num_clients} clients | batch={batch_size}"
    )

    for round_id in range(1, num_rounds + 1):
        if torch.cuda.is_available():
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
            ).to(device)
            set_params(local_model, global_params)
            payload_events.append(
                {
                    "round": round_id,
                    "client_id": client_id,
                    "direction": "downlink",
                    "payload_type": "model",
                    **download_payload,
                    "error_norm": 0.0,
                }
            )

            params, client_train_loss = local_train(
                model=local_model,
                loader=client_trainloaders[client_id],
                epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
            )
            upload_payload = summarize_payload(params)
            payload_events.append(
                {
                    "round": round_id,
                    "client_id": client_id,
                    "direction": "uplink",
                    "payload_type": "update",
                    **upload_payload,
                    "error_norm": 0.0,
                }
            )
            client_params_list.append(params)
            client_sizes.append(len(client_y_list[client_id]))
            client_train_losses.append(client_train_loss)

        local_time_sec = time.perf_counter() - local_t0
        server_t0 = time.perf_counter()
        averaged_params = fed_avg(client_params_list, client_sizes)
        set_params(global_model, averaged_params)
        metrics = evaluate(global_model, global_testloader, device, num_classes)
        server_time_sec = time.perf_counter() - server_t0
        train_loss = float(np.mean(client_train_losses))
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if torch.cuda.is_available()
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
                "train_loss": train_loss,
                "local_time_sec": round(local_time_sec, 2),
                "server_time_sec": round(server_time_sec, 2),
                "peak_vram_mb": round(float(peak_vram_mb), 2),
            }
        )

        checkpoint_path = checkpoint_dir / f"round_{round_id:02d}_fedavg.pt"
        torch.save(
            {
                name: tensor.detach().cpu()
                for name, tensor in global_model.state_dict().items()
            },
            checkpoint_path,
        )
        print(f"--> Checkpoint saved to {checkpoint_path}")
        if round_id > 1:
            previous_checkpoint = checkpoint_dir / f"round_{round_id - 1:02d}_fedavg.pt"
            if previous_checkpoint.exists():
                try:
                    previous_checkpoint.unlink()
                except Exception:
                    pass

        print(
            f"Round {round_id:02d}/{num_rounds} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1_W={metrics['f1_weighted']:.4f} | "
            f"F1_M={metrics['f1_macro']:.4f} | "
            f"Local={local_time_sec:.1f}s | Server={server_time_sec:.1f}s"
        )

    return global_model, metrics_log, payload_events


def main() -> int:
    num_clients = NUM_CLIENTS
    data_dir = resolve_cicandmal_data_dir(
        num_clients,
        project_root=PROJECT_ROOT,
    )
    output_dir = resolve_output_dir("fedavg", num_clients, PROJECT_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Using data from: {data_dir}")
    set_seed(SEED)

    metadata, client_X_list, client_y_list, X_test, y_test = load_cicandmal_data(
        data_dir,
        expected_clients=num_clients,
    )
    classes = metadata["classes"]
    num_features = int(metadata["num_features"])
    num_classes = int(metadata["num_classes"])
    print(f"Classes: {classes}")
    print(f"Number of features: {num_features}")
    print(f"Number of clients: {num_clients}")
    print(f"Number of classes: {num_classes}")
    print(f"Global Test Data: X_test = {X_test.shape}, y_test = {y_test.shape}")
    for client_id, y_client in enumerate(client_y_list):
        print(
            f"Client {client_id}: samples = {len(y_client)}, "
            f"shape = {client_X_list[client_id].shape}"
        )

    client_trainloaders, global_testloader = make_client_loaders(
        client_X_list,
        client_y_list,
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
    )
    global_model, metrics_log, payload_events = train_federated(
        client_trainloaders=client_trainloaders,
        client_y_list=client_y_list,
        num_clients=num_clients,
        global_testloader=global_testloader,
        num_features=num_features,
        num_classes=num_classes,
        device=device,
        num_rounds=NUM_ROUNDS,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=output_dir / "checkpoints",
    )
    save_results(
        output_dir=output_dir,
        classes=classes,
        num_classes=num_classes,
        num_rounds=NUM_ROUNDS,
        metrics_log=metrics_log,
        payload_events=payload_events,
        global_model=global_model,
        global_testloader=global_testloader,
        device=device,
        method_name="FedAvg",
    )
    archive_results(
        output_dir,
        f"FedAVG_{num_clients}clients_results.zip",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
