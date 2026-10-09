"""FedDST dynamic sparse baseline ported from the 10/20/50-client notebooks."""

from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import io
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_NAME = "FedDST"
METHOD_SLUG = "feddst"
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

SPARSITY = 0.50
MASK_UPDATE_INTERVAL = 15
MASK_UPDATE_END_ROUND = NUM_ROUNDS
MASK_UPDATE_ALPHA = 0.05
MASK_UPDATE_EPOCH = 1


def clone_masks(masks):
    return {k: v.detach().clone() for k, v in masks.items()}


def count_mask_stats(masks):
    # Sparsity is defined over ERK-prunable matrix weights only.
    # Biases/LayerNorm vectors remain dense and are excluded consistently.
    prunable_masks = [m for m in masks.values() if m.ndim == 2]
    active = sum(int(m.sum().item()) for m in prunable_masks)
    total = sum(m.numel() for m in prunable_masks)
    return active, total, 1.0 - active / max(total, 1)


def apply_masks(model, masks):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(device=param.device, dtype=param.dtype))


def summarize_sparse_payload(state_dict, masks):
    cpu_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }
    cpu_masks = {
        name: mask.detach().cpu().to(dtype=torch.bool).contiguous()
        for name, mask in masks.items()
        if torch.is_tensor(mask)
    }
    compact_values = {}
    compact_masks = {}
    total_params = 0
    nnz = 0
    dense_values_bytes = 0
    values_bytes = 0
    for name, tensor in cpu_state.items():
        active_mask = cpu_masks.get(name, torch.ones_like(tensor, dtype=torch.bool))
        active_mask = active_mask.reshape(tensor.shape)
        compact_values[name] = tensor[active_mask].contiguous()
        total_params += int(tensor.numel())
        nnz += int(active_mask.sum().item())
        dense_values_bytes += int(tensor.numel() * tensor.element_size())
        values_bytes += int(compact_values[name].numel() * tensor.element_size())
    mask_elements = 0
    for name, mask in cpu_masks.items():
        bits = np.packbits(mask.flatten().numpy().astype(np.uint8))
        compact_masks[name] = torch.from_numpy(bits.copy())
        mask_elements += int(mask.numel())
    mask_bytes = int((mask_elements + 7) // 8)
    compact_payload = {"values": compact_values, "mask_bits": compact_masks}
    buffer = io.BytesIO()
    torch.save(compact_payload, buffer)
    serialized_bytes = int(buffer.getbuffer().nbytes)
    metadata_bytes = max(serialized_bytes - values_bytes - mask_bytes, 0)
    return {
        "raw_bytes": int(dense_values_bytes + mask_bytes),
        "serialized_bytes": serialized_bytes,
        "values_bytes": values_bytes,
        "indices_bytes": 0,
        "scales_bytes": 0,
        "mask_bytes": mask_bytes,
        "prototype_bytes": 0,
        "metadata_bytes": metadata_bytes,
        "nnz": nnz,
        "total_params": total_params,
        "sparsity": float(1.0 - nnz / max(total_params, 1)),
    }


def get_erk_sparsities(model, target_sparsity):
    layers = [
        (name, param.numel(), (param.shape[1] + param.shape[0]) / float(param.numel()))
        for name, param in model.named_parameters()
        if param.requires_grad and param.ndim == 2
    ]
    if not layers:
        return {}

    target_active = (1.0 - target_sparsity) * sum(n for _, n, _ in layers)
    remaining, density = list(layers), {}

    while remaining:
        denom = sum(n * s for _, n, s in remaining)
        c = target_active / denom if denom else 0.0
        saturated = [(nm, n, s) for nm, n, s in remaining if c * s >= 1.0]
        if not saturated:
            for nm, _, s in remaining:
                density[nm] = max(0.0, min(1.0, c * s))
            break
        for nm, _, _ in saturated:
            density[nm] = 1.0
        target_active -= sum(n for _, n, _ in saturated)
        remaining = [x for x in remaining if x not in saturated]
        if target_active <= 0:
            for nm, _, _ in remaining:
                density[nm] = 0.0
            break

    return {nm: 1.0 - d for nm, d in density.items()}


def initialize_masks(model, target_sparsity=None, layer_sparsities=None):
    if target_sparsity is None:
        target_sparsity = SPARSITY
    set_seed(SEED)
    if layer_sparsities is None:
        layer_sparsities = get_erk_sparsities(model, target_sparsity)
    masks = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and name in layer_sparsities:
            s = layer_sparsities[name]
            n = param.numel()
            k = max(1, int(round((1.0 - s) * n)))
            mask = torch.zeros(n, device=param.device, dtype=param.dtype)
            mask[torch.randperm(n, device=param.device)[:k]] = 1.0
            masks[name] = mask.view_as(param)
        else:
            masks[name] = torch.ones_like(param)
    apply_masks(model, masks)
    return masks


def cosine_alpha(round_id):
    if MASK_UPDATE_END_ROUND <= 1:
        return float(MASK_UPDATE_ALPHA)
    if round_id >= MASK_UPDATE_END_ROUND:
        return 0.0
    return float(
        (MASK_UPDATE_ALPHA / 2.0)
        * (1.0 + math.cos(((round_id - 1) * math.pi) / MASK_UPDATE_END_ROUND))
    )


def prune_and_grow(model, masks, layer_sparsities, alpha_r):
    new_masks = {}
    for name, param in model.named_parameters():
        if name not in masks:
            continue

        # Only ERK-prunable matrix weights are rewired. Biases and
        # normalization parameters stay dense, matching initialization.
        if name not in layer_sparsities:
            new_masks[name] = torch.ones_like(param)
            continue

        old_mask = masks[name].to(param.device)
        n = param.numel()
        base_sparsity = float(layer_sparsities[name])
        final_k = max(1, int(round((1.0 - base_sparsity) * n)))
        temp_s = base_sparsity + (1.0 - base_sparsity) * alpha_r
        temp_k = max(1, min(final_k, int(round((1.0 - temp_s) * n))))
        grow_k = final_k - temp_k

        # 1. Magnitude Prune: Keep top-temp_k |w| among active connections
        scores = param.detach().abs().flatten()
        scores[old_mask.flatten() <= 0] = -1.0
        keep_idx = torch.topk(scores, k=temp_k, largest=True, sorted=False).indices
        new_mask = torch.zeros(n, device=param.device, dtype=param.dtype)
        new_mask[keep_idx] = 1.0

        # 2. Gradient-Magnitude Grow: Select top-grow_k |∇L| from inactive connections
        if grow_k > 0:
            if param.grad is not None:
                grad = param.grad.detach().abs().flatten().clone()
            else:
                grad = torch.rand(n, device=param.device)
            grad[new_mask > 0] = -1.0
            grow_idx = torch.topk(grad, k=grow_k, largest=True, sorted=False).indices
            new_mask[grow_idx] = 1.0
            with torch.no_grad():
                param.data.flatten()[grow_idx] = 0.0

        new_masks[name] = new_mask.view_as(param)

    apply_masks(model, new_masks)
    return new_masks


def sparse_weighted_average(client_params, client_masks, client_sizes, server_params):
    total = float(sum(client_sizes))
    avg = OrderedDict()
    mask_keys = set(client_masks[0].keys())

    for key in client_params[0]:
        first = client_params[0][key]
        if not torch.is_floating_point(first):
            avg[key] = first.clone()
            continue
        if key in mask_keys:
            num = torch.zeros_like(first, dtype=torch.float32).cpu()
            den = torch.zeros_like(first, dtype=torch.float32).cpu()
            for p, m, sz in zip(client_params, client_masks, client_sizes):
                w = p[key].detach().float().cpu()
                mk = m[key].detach().float().cpu()
                num += sz * w * mk
                den += sz * mk
            old = server_params[key].detach().float().cpu()
            avg[key] = torch.where(den > 0, num / den.clamp(min=1e-12), old).to(
                dtype=first.dtype
            )
        else:
            t = torch.zeros_like(first, dtype=torch.float32).cpu()
            for p, sz in zip(client_params, client_sizes):
                t += p[key].detach().float().cpu() * (sz / total)
            avg[key] = t.to(dtype=first.dtype)

    return avg


def server_magnitude_prune(model, layer_sparsities):
    masks = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Keep non-prunable parameters in the mask dictionary as dense so
        # the mask set is identical before and after server reallocation.
        if param.ndim != 2 or name not in layer_sparsities:
            masks[name] = torch.ones_like(param)
            continue

        flat = param.detach().abs().flatten()
        n = flat.numel()
        layer_sparsity = float(layer_sparsities[name])
        k = max(1, int(round((1.0 - layer_sparsity) * n)))
        top_i = torch.topk(flat, k=min(k, n), largest=True, sorted=False).indices
        mask = torch.zeros_like(flat)
        mask[top_i] = 1.0
        masks[name] = mask.view_as(param)
    return masks


def train_one_client(
    model,
    loader,
    masks,
    round_id,
    layer_sparsities,
    *,
    num_epochs,
    learning_rate,
    device,
):
    model.train()
    apply_masks(model, masks)
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    alpha_r = cosine_alpha(round_id)

    total_loss, total_correct, total_seen = 0.0, 0, 0
    mask_changed = False

    for epoch in range(1, num_epochs + 1):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            apply_masks(model, masks)

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_correct += (preds == yb).sum().item()
                total_seen += yb.size(0)
                total_loss += loss.item() * yb.size(0)

        if (
            round_id % MASK_UPDATE_INTERVAL == 0
            and epoch == MASK_UPDATE_EPOCH
            and round_id < MASK_UPDATE_END_ROUND
            and alpha_r > 0
        ):
            masks = prune_and_grow(model, masks, layer_sparsities, alpha_r)
            mask_changed = True

    _, _, cur_sparsity = count_mask_stats(masks)
    return (
        get_params(model),
        clone_masks(masks),
        {
            "loss": total_loss / max(total_seen, 1),
            "acc": total_correct / max(total_seen, 1),
            "samples": len(loader.dataset),
            "mask_changed": mask_changed,
            "sparsity": cur_sparsity,
        },
    )


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
    auxiliary_dir = checkpoint_dir / "auxiliary"
    mask_checkpoint_dir = auxiliary_dir / "masks"
    mask_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the notebook's model initialization diagnostic, then reset the
    # training RNG exactly where its federated cell resets it.
    test_model = build_model(num_features, num_classes, device)
    dummy_in = torch.randn(4, 1, num_features).to(device)
    with torch.no_grad():
        dummy_out = test_model(dummy_in)
    print(
        f"Model initialized successfully: input {tuple(dummy_in.shape)} -> output {tuple(dummy_out.shape)}"
    )
    del test_model, dummy_in, dummy_out
    set_seed(SEED)
    global_model = build_model(num_features, num_classes, device)
    layer_sparsities = get_erk_sparsities(global_model, SPARSITY)
    global_masks = initialize_masks(global_model, SPARSITY, layer_sparsities)
    active0, total0, s0 = count_mask_stats(global_masks)
    torch.save(layer_sparsities, auxiliary_dir / "feddst_layer_sparsities.pt")
    print(f"Initial Sparsity: {s0:.4f} (Active {active0}/{total0} prunable params)")
    metrics_log, payload_events, client_diagnostics = [], [], []

    for rnd in range(1, num_rounds + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        server_params = get_params(global_model)
        server_masks = clone_masks(global_masks)
        download_payload = summarize_sparse_payload(server_params, server_masks)
        alpha_r = cosine_alpha(rnd)
        client_params_list, client_masks_list, client_sizes_rnd = [], [], []
        local_losses, local_accs = [], []
        mask_changed_count = 0
        selected_clients = list(range(num_clients))
        local_t0 = time.perf_counter()

        for cid in selected_clients:
            local_model = build_model(num_features, num_classes, device)
            set_params(local_model, server_params)
            local_masks = clone_masks(server_masks)
            apply_masks(local_model, local_masks)
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "downlink",
                    "sparse_model",
                    download_payload,
                )
            )
            lp, lm, info = train_one_client(
                local_model,
                client_trainloaders[cid],
                local_masks,
                rnd,
                layer_sparsities,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                device=device,
            )
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "uplink",
                    "sparse_update",
                    summarize_sparse_payload(lp, lm),
                )
            )
            client_params_list.append(lp)
            client_masks_list.append(lm)
            client_sizes_rnd.append(info["samples"])
            local_losses.append(info["loss"])
            local_accs.append(info["acc"])
            mask_changed_count += int(info["mask_changed"])
            client_diagnostics.append({"round": rnd, "client_id": cid, **info})
            del local_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        local_time_sec = time.perf_counter() - local_t0
        server_t0 = time.perf_counter()
        agg_params = sparse_weighted_average(
            client_params_list,
            client_masks_list,
            client_sizes_rnd,
            server_params,
        )
        set_params(global_model, agg_params)
        global_masks = server_magnitude_prune(global_model, layer_sparsities)
        apply_masks(global_model, global_masks)
        eval_res = evaluate(
            global_model,
            global_testloader,
            device,
            num_classes,
            include_loss=True,
            include_mcc=True,
        )
        active, total_m, actual_s = count_mask_stats(global_masks)
        server_time_sec = time.perf_counter() - server_t0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        round_metrics = {
            "round": rnd,
            **{
                key: value
                for key, value in eval_res.items()
                if key not in {"preds", "labels"}
            },
            "train_loss": float(np.mean(local_losses)),
            "local_time_sec": round(local_time_sec, 2),
            "server_time_sec": round(server_time_sec, 2),
            "peak_vram_mb": round(float(peak_vram_mb), 2),
            "sparsity_actual": actual_s,
            "alpha_r": alpha_r,
            "mask_changed_count": mask_changed_count,
        }
        metrics_log.append(round_metrics)
        ckpt_path = checkpoint_dir / f"round_{rnd:02d}_feddst.pt"
        save_model_checkpoint(global_model.state_dict(), ckpt_path)
        cpu_masks = {name: mask.detach().cpu() for name, mask in global_masks.items()}
        torch.save(cpu_masks, mask_checkpoint_dir / f"round_{rnd:02d}_feddst_masks.pt")
        print(
            f"Round {rnd:02d}/{num_rounds:02d} | Loss={eval_res['loss']:.4f} | "
            f"Acc={eval_res['accuracy']:.4f} | F1_M={eval_res['f1_macro']:.4f} | "
            f"F1_W={eval_res['f1_weighted']:.4f} | S={actual_s:.4f} | "
            f"MaskChg={mask_changed_count}/{num_clients} | "
            f"Local={local_time_sec:.1f}s | Server={server_time_sec:.1f}s"
        )

    results_df = pd.DataFrame(metrics_log)
    best_round = results_df.loc[results_df["f1_macro"].idxmax()]
    last_round = results_df.iloc[-1]
    summary = pd.DataFrame(
        [
            {
                "method": METHOD_NAME,
                "best_round": int(best_round["round"]),
                "best_accuracy": best_round["accuracy"],
                "best_f1_macro": best_round["f1_macro"],
                "best_f1_weighted": best_round["f1_weighted"],
                "last_round": int(last_round["round"]),
                "last_accuracy": last_round["accuracy"],
                "last_f1_macro": last_round["f1_macro"],
                "last_f1_weighted": last_round["f1_weighted"],
                "sparsity_target": SPARSITY,
                "sparsity_actual": best_round["sparsity_actual"],
            }
        ]
    )

    return {
        "model": global_model,
        "metrics_log": metrics_log,
        "payload_events": payload_events,
        "masks": global_masks,
        "layer_sparsities": layer_sparsities,
        "client_diagnostics": client_diagnostics,
        "initial_mask_stats": {"active": active0, "total": total0, "sparsity": s0},
        "summary": summary,
        "extra_tables": {},
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
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_federated(
        client_trainloaders=data["client_trainloaders"],
        client_y_list=data["client_y_list"],
        num_clients=data["num_clients"],
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
        include_classification_report=True,
        extra_tables=result["extra_tables"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
