"""FedGPD port of Distillation/FedGPD/{10,20,50} clients notebooks.

The classwise distribution loss acts on the post-ReLU dense features, including
round one's zero global prototypes. Model weights and class-support-weighted
prototypes are aggregated separately; neither is a sparse communication codec.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

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
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_NAME = "FedGPD"
METHOD_SLUG = "fedgpd"
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
FEDGPD_LAMBDA = 0.05
TEMPERATURE = 2.0


def summarize_model_prototype_payload(
    state_dict, prototype_payload=None, class_counts=None
):
    """Measure the source's dense weights/prototypes/support torch.save envelope."""
    cpu_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }
    if torch.is_tensor(prototype_payload):
        cpu_prototypes = prototype_payload.detach().cpu().contiguous()
        payload_prototypes = cpu_prototypes
        prototype_tensors = [cpu_prototypes]
    elif isinstance(prototype_payload, dict):
        payload_prototypes = {
            int(key): value.detach().cpu().contiguous()
            for key, value in prototype_payload.items()
        }
        prototype_tensors = list(payload_prototypes.values())
    else:
        payload_prototypes = None
        prototype_tensors = []
    prototype_bytes = int(sum(t.numel() * t.element_size() for t in prototype_tensors))
    payload = {"state_dict": cpu_state, "prototypes": payload_prototypes}
    if class_counts is not None:
        payload["class_counts"] = {
            int(key): int(value) for key, value in class_counts.items()
        }
    values_bytes = int(sum(t.numel() * t.element_size() for t in cpu_state.values()))
    total_params = int(
        sum(t.numel() for t in cpu_state.values())
        + sum(t.numel() for t in prototype_tensors)
    )
    nnz = int(
        sum(torch.count_nonzero(t).item() for t in cpu_state.values())
        + sum(torch.count_nonzero(t).item() for t in prototype_tensors)
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    serialized_bytes = int(buffer.getbuffer().nbytes)
    return {
        "raw_bytes": values_bytes + prototype_bytes,
        "serialized_bytes": serialized_bytes,
        "values_bytes": values_bytes,
        "indices_bytes": 0,
        "scales_bytes": 0,
        "mask_bytes": 0,
        "prototype_bytes": prototype_bytes,
        "metadata_bytes": max(serialized_bytes - values_bytes - prototype_bytes, 0),
        "nnz": nnz,
        "total_params": total_params,
        "sparsity": float(1.0 - nnz / max(total_params, 1)),
    }


def local_train(model, loader, global_protos, epochs, lr, device, lam, T):
    model.train()
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    present_classes = torch.unique(loader.dataset.tensors[1]).cpu().numpy()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, z = model(xb, return_features=True)
            loss_ce = criterion(logits, yb)
            loss_gpd = 0.0
            if global_protos is not None:
                for k in present_classes:
                    mask_k = yb == k
                    if mask_k.sum() > 0:
                        local_proto = z[mask_k].mean(dim=0)
                        p_local = torch.log_softmax(local_proto / T, dim=0)
                        p_global = torch.softmax(global_protos[k].to(device) / T, dim=0)
                        loss_gpd += -torch.sum(p_global * p_local)
            # This is prototype-distribution cross entropy, not T-squared KL.
            loss = loss_ce + lam * loss_gpd
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item())
            num_batches += 1
    return get_params(model), running_loss / max(num_batches, 1)


def compute_local_prototypes(model, loader, device):
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            _, z = model(xb.to(device), return_features=True)
            features_list.append(z.cpu())
            labels_list.append(yb)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    local_protos, class_counts = {}, {}
    for k in torch.unique(labels).numpy():
        mask = labels == k
        local_protos[int(k)] = features[mask].mean(dim=0)
        class_counts[int(k)] = int(mask.sum())
    return local_protos, class_counts


def aggregate_prototypes(
    client_protos_list, client_counts_list, num_classes, feature_dim
):
    global_protos = torch.zeros((num_classes, feature_dim))
    for k in range(num_classes):
        sum_proto = torch.zeros(feature_dim)
        total_count = 0
        for i in range(len(client_protos_list)):
            if k in client_protos_list[i]:
                sum_proto += client_protos_list[i][k] * client_counts_list[i][k]
                total_count += client_counts_list[i][k]
        if total_count > 0:
            global_protos[k] = sum_proto / total_count
    return global_protos


def train_federated(
    *,
    client_trainloaders,
    client_y_list,
    num_clients,
    global_testloader,
    num_features,
    num_classes,
    device,
    num_rounds,
    num_epochs,
    batch_size,
    learning_rate,
    checkpoint_dir,
):
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prototype_checkpoint_dir = checkpoint_dir / "auxiliary" / "prototypes"
    prototype_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    global_model = HybridBLSTM_GRU(
        input_size=num_features,
        num_classes=num_classes,
        blstm_hidden=BLSTM_HIDDEN,
        gru_hidden=GRU_HIDDEN,
        dense_hidden=DENSE_HIDDEN,
        dropout=DROPOUT,
    ).to(device)
    global_protos = torch.zeros((num_classes, DENSE_HIDDEN))
    metrics_log, payload_events = [], []
    client_protos_list, client_counts_list = [], []
    client_train_losses = []
    metrics = None
    print(
        f"Starting {METHOD_NAME}: {num_rounds} rounds | {num_clients} clients | batch={batch_size}"
    )

    for rnd in range(1, num_rounds + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        global_params = get_params(global_model)
        download_payload = summarize_model_prototype_payload(
            global_params, global_protos
        )
        local_t0 = time.perf_counter()
        client_params_list, client_sizes, client_train_losses = [], [], []
        client_protos_list, client_counts_list = [], []
        for cid in range(num_clients):
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
                    rnd,
                    cid,
                    "downlink",
                    "model+prototype",
                    download_payload,
                )
            )
            params, client_train_loss = local_train(
                model=local_model,
                loader=client_trainloaders[cid],
                global_protos=global_protos,
                epochs=num_epochs,
                lr=learning_rate,
                device=device,
                lam=FEDGPD_LAMBDA,
                T=TEMPERATURE,
            )
            client_params_list.append(params)
            client_sizes.append(len(client_y_list[cid]))
            client_train_losses.append(client_train_loss)
            local_protos, class_counts = compute_local_prototypes(
                local_model,
                client_trainloaders[cid],
                device,
            )
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "uplink",
                    "update+prototype",
                    summarize_model_prototype_payload(
                        params, local_protos, class_counts
                    ),
                )
            )
            client_protos_list.append(local_protos)
            client_counts_list.append(class_counts)
        local_time_sec = time.perf_counter() - local_t0

        server_t0 = time.perf_counter()
        averaged_params = fed_avg(client_params_list, client_sizes)
        set_params(global_model, averaged_params)
        global_protos = aggregate_prototypes(
            client_protos_list,
            client_counts_list,
            num_classes,
            DENSE_HIDDEN,
        )
        metrics = evaluate(global_model, global_testloader, device, num_classes)
        server_time_sec = time.perf_counter() - server_t0
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
                    for key, value in metrics.items()
                    if key not in ("preds", "labels")
                },
                "train_loss": float(np.mean(client_train_losses)),
                "local_time_sec": round(local_time_sec, 2),
                "server_time_sec": round(server_time_sec, 2),
                "peak_vram_mb": round(float(peak_vram_mb), 2),
            }
        )
        ckpt_path = checkpoint_dir / f"round_{rnd:02d}_fedgpd.pt"
        save_model_checkpoint(averaged_params, ckpt_path)
        torch.save(
            global_protos.detach().cpu(),
            prototype_checkpoint_dir / f"round_{rnd:02d}_fedgpd_prototypes.pt",
        )
        print(f"--> Checkpoint saved to {ckpt_path}")
        print(
            f"Round {rnd:02d}/{num_rounds} | Acc={metrics['accuracy']:.4f} | "
            f"F1_W={metrics['f1_weighted']:.4f} | F1_M={metrics['f1_macro']:.4f} | "
            f"Local={local_time_sec:.1f}s | Server={server_time_sec:.1f}s"
        )
    return {
        "model": global_model,
        "metrics_log": metrics_log,
        "payload_events": payload_events,
        "prototypes": global_protos,
        "client_prototypes": client_protos_list,
        "client_class_counts": client_counts_list,
        "client_train_losses": client_train_losses,
        "final_metrics": metrics,
        "prototype_checkpoint_dir": prototype_checkpoint_dir,
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
