"""ZeroShot DepGraph-style structural baseline from the 10/20/50-client notebooks."""

from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import copy
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
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
    summarize_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_NAME = "ZeroShot"
METHOD_SLUG = "zeroshot"
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

PRUNING_RATE = 0.50
CLIENT_FRACTION = 1.0
INCLUDE_BIAS_IN_SCORE = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LSTM, nn.GRU)):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param.data, nonlinearity="relu")
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)


class DepGraphPrunePlan:
    def __init__(
        self,
        pruning_rate,
        keep_lstm_fwd,
        keep_lstm_bwd,
        keep_gru,
        keep_fc_hidden,
        raw_scores,
        normalized_scores,
    ):
        self.pruning_rate = pruning_rate
        self.keep_lstm_fwd = keep_lstm_fwd
        self.keep_lstm_bwd = keep_lstm_bwd
        self.keep_gru = keep_gru
        self.keep_fc_hidden = keep_fc_hidden
        self.raw_scores = raw_scores
        self.normalized_scores = normalized_scores
        self.new_lstm_hidden = len(keep_lstm_fwd)
        self.new_gru_hidden = len(keep_gru)
        self.new_fc_hidden = len(keep_fc_hidden)

    def summary(self):
        return {
            "pruning_rate": self.pruning_rate,
            "new_lstm_hidden": self.new_lstm_hidden,
            "new_gru_hidden": self.new_gru_hidden,
            "new_fc_hidden": self.new_fc_hidden,
        }


def _l1_squared(tensor):
    # Lightweight-Fed-NIDS Eq. 7: ||theta[k]||_1^2.
    return torch.sum(torch.abs(tensor)) ** 2


def _group_l1_squared(tensor, row_indices=None, col_indices=None):
    # Take the union of elements in the dependency group to avoid double-counting
    # the intersection between the row and column of weight_hh.
    if tensor.ndim == 1:
        indices = row_indices if row_indices is not None else col_indices
        if indices is None or len(indices) == 0:
            return tensor.new_zeros(())
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask[indices] = True
    elif tensor.ndim == 2:
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        if row_indices is not None and len(row_indices) > 0:
            mask[row_indices, :] = True
        if col_indices is not None and len(col_indices) > 0:
            mask[:, col_indices] = True
    else:
        raise ValueError(f"Unsupported tensor rank in dependency group: {tensor.ndim}")
    return _l1_squared(tensor[mask])


def _as_index_tensor(indices, dev):
    return torch.tensor(indices, dtype=torch.long, device=dev)


def _gate_indices(unit_indices, hidden_size, num_gates, dev):
    # Used for LSTM (4 gates: i, f, g, o) or GRU (3 gates: r, z, n)
    indices = []
    for u in unit_indices:
        for g in range(num_gates):
            indices.append(u + g * hidden_size)
    return torch.tensor(indices, dtype=torch.long, device=dev)


def _normalize_scores_like_paper(scores, top_n=None):
    # Normalize importance according to Lightweight-Fed-NIDS Eq. 7
    scores = scores.float()
    if top_n is None:
        top_n = len(scores)
    top_vals, _ = torch.topk(scores, min(top_n, len(scores)))
    return (len(scores) * scores) / (top_vals.sum() + 1e-9)


def _select_keep_indices(norm_scores, pruning_rate):
    num_keep = int(len(norm_scores) * (1.0 - pruning_rate))
    num_keep = max(1, num_keep)  # Keep at least one neuron
    _, indices = torch.topk(norm_scores, num_keep)
    return sorted(indices.tolist())


def compute_depgraph_group_scores(model, include_bias=True):
    old_sd = model.state_dict()
    dev = next(model.parameters()).device
    H_lstm, H_gru, H_fc = model.lstm_hidden, model.gru_hidden, model.fc_hidden
    raw = {}

    # 1. BLSTM Forward group scores
    scores_fwd = []
    w_ih, w_hh, gru_w_ih = (
        old_sd["blstm.weight_ih_l0"].to(dev),
        old_sd["blstm.weight_hh_l0"].to(dev),
        old_sd["gru.weight_ih_l0"].to(dev),
    )
    b_ih, b_hh = old_sd["blstm.bias_ih_l0"].to(dev), old_sd["blstm.bias_hh_l0"].to(dev)
    ln_blstm_w, ln_blstm_b = (
        old_sd["ln_blstm.weight"].to(dev),
        old_sd["ln_blstm.bias"].to(dev),
    )
    for k in range(H_lstm):
        rows = _gate_indices([k], H_lstm, 4, dev)
        unit = _as_index_tensor([k], dev)
        score = _group_l1_squared(w_ih, row_indices=rows)
        score += _group_l1_squared(w_hh, row_indices=rows, col_indices=unit)
        score += _group_l1_squared(gru_w_ih, col_indices=unit)
        score += _group_l1_squared(ln_blstm_w, row_indices=unit)
        if include_bias:
            score += _group_l1_squared(b_ih, row_indices=rows)
            score += _group_l1_squared(b_hh, row_indices=rows)
            score += _group_l1_squared(ln_blstm_b, row_indices=unit)
        scores_fwd.append(score.cpu())
    raw["lstm_fwd"] = torch.stack(scores_fwd).float()

    # 2. BLSTM Backward group scores
    scores_bwd = []
    w_ih_r, w_hh_r = (
        old_sd["blstm.weight_ih_l0_reverse"].to(dev),
        old_sd["blstm.weight_hh_l0_reverse"].to(dev),
    )
    b_ih_r, b_hh_r = (
        old_sd["blstm.bias_ih_l0_reverse"].to(dev),
        old_sd["blstm.bias_hh_l0_reverse"].to(dev),
    )
    for k in range(H_lstm):
        rows = _gate_indices([k], H_lstm, 4, dev)
        unit = _as_index_tensor([H_lstm + k], dev)
        recurrent_unit = _as_index_tensor([k], dev)
        score = _group_l1_squared(w_ih_r, row_indices=rows)
        score += _group_l1_squared(w_hh_r, row_indices=rows, col_indices=recurrent_unit)
        score += _group_l1_squared(gru_w_ih, col_indices=unit)
        score += _group_l1_squared(ln_blstm_w, row_indices=unit)
        if include_bias:
            score += _group_l1_squared(b_ih_r, row_indices=rows)
            score += _group_l1_squared(b_hh_r, row_indices=rows)
            score += _group_l1_squared(ln_blstm_b, row_indices=unit)
        scores_bwd.append(score.cpu())
    raw["lstm_bwd"] = torch.stack(scores_bwd).float()

    # 3. GRU group scores
    scores_gru = []
    gru_w_hh = old_sd["gru.weight_hh_l0"].to(dev)
    fc1_w = old_sd["fc1.weight"].to(dev)
    b_gru_ih, b_gru_hh = (
        old_sd["gru.bias_ih_l0"].to(dev),
        old_sd["gru.bias_hh_l0"].to(dev),
    )
    ln_gru_w, ln_gru_b = old_sd["ln_gru.weight"].to(dev), old_sd["ln_gru.bias"].to(dev)
    for k in range(H_gru):
        rows = _gate_indices([k], H_gru, 3, dev)
        unit = _as_index_tensor([k], dev)
        score = _group_l1_squared(gru_w_ih, row_indices=rows)
        score += _group_l1_squared(gru_w_hh, row_indices=rows, col_indices=unit)
        score += _group_l1_squared(ln_gru_w, row_indices=unit)
        score += _group_l1_squared(fc1_w, col_indices=unit)
        if include_bias:
            score += _group_l1_squared(b_gru_ih, row_indices=rows)
            score += _group_l1_squared(b_gru_hh, row_indices=rows)
            score += _group_l1_squared(ln_gru_b, row_indices=unit)
        scores_gru.append(score.cpu())
    raw["gru"] = torch.stack(scores_gru).float()

    # 4. Dense FC group scores
    scores_fc = []
    fc1_w, fc1_b, fc_out_w = (
        old_sd["fc1.weight"].to(dev),
        old_sd["fc1.bias"].to(dev),
        old_sd["fc_out.weight"].to(dev),
    )
    ln_fc_w, ln_fc_b = old_sd["ln_fc.weight"].to(dev), old_sd["ln_fc.bias"].to(dev)
    for k in range(H_fc):
        unit = _as_index_tensor([k], dev)
        score = _group_l1_squared(fc1_w, row_indices=unit)
        score += _group_l1_squared(fc_out_w, col_indices=unit)
        score += _group_l1_squared(ln_fc_w, row_indices=unit)
        if include_bias:
            score += _group_l1_squared(fc1_b, row_indices=unit)
            score += _group_l1_squared(ln_fc_b, row_indices=unit)
        scores_fc.append(score.cpu())
    raw["fc_hidden"] = torch.stack(scores_fc).float()

    normalized = {
        name: _normalize_scores_like_paper(scores) for name, scores in raw.items()
    }
    return raw, normalized


def build_zero_shot_depgraph_plan(model, pruning_rate, include_bias_in_score=True):
    raw, normalized = compute_depgraph_group_scores(
        model, include_bias=include_bias_in_score
    )
    keep_lstm_fwd = _select_keep_indices(normalized["lstm_fwd"], pruning_rate)
    keep_lstm_bwd = _select_keep_indices(normalized["lstm_bwd"], pruning_rate)
    n = min(len(keep_lstm_fwd), len(keep_lstm_bwd))
    return DepGraphPrunePlan(
        pruning_rate=pruning_rate,
        keep_lstm_fwd=keep_lstm_fwd[:n],
        keep_lstm_bwd=keep_lstm_bwd[:n],
        keep_gru=_select_keep_indices(normalized["gru"], pruning_rate),
        keep_fc_hidden=_select_keep_indices(normalized["fc_hidden"], pruning_rate),
        raw_scores=raw,
        normalized_scores=normalized,
    )


def _copy_lstm_direction(old_sd, new_sd, old_hidden, new_hidden, keep_units, suffix):
    old_rows = _gate_indices(keep_units, old_hidden, 4, torch.device("cpu"))
    new_rows = _gate_indices(range(new_hidden), new_hidden, 4, torch.device("cpu"))
    keep_t = torch.tensor(keep_units, dtype=torch.long)
    new_sd[f"blstm.weight_ih_l0{suffix}"][new_rows, :] = old_sd[
        f"blstm.weight_ih_l0{suffix}"
    ].index_select(0, old_rows)
    new_sd[f"blstm.weight_hh_l0{suffix}"][new_rows, :] = (
        old_sd[f"blstm.weight_hh_l0{suffix}"]
        .index_select(0, old_rows)
        .index_select(1, keep_t)
    )
    new_sd[f"blstm.bias_ih_l0{suffix}"][new_rows] = old_sd[
        f"blstm.bias_ih_l0{suffix}"
    ].index_select(0, old_rows)
    new_sd[f"blstm.bias_hh_l0{suffix}"][new_rows] = old_sd[
        f"blstm.bias_hh_l0{suffix}"
    ].index_select(0, old_rows)


def _copy_gru(
    old_sd,
    new_sd,
    old_gru_hidden,
    new_gru_hidden,
    keep_gru,
    old_lstm_hidden,
    keep_lstm_fwd,
    keep_lstm_bwd,
):
    old_rows = _gate_indices(keep_gru, old_gru_hidden, 3, torch.device("cpu"))
    new_rows = _gate_indices(
        range(new_gru_hidden), new_gru_hidden, 3, torch.device("cpu")
    )
    keep_gru_t = torch.tensor(keep_gru, dtype=torch.long)
    old_gru_input_cols = torch.tensor(
        list(keep_lstm_fwd) + [old_lstm_hidden + i for i in keep_lstm_bwd],
        dtype=torch.long,
    )
    new_sd["gru.weight_ih_l0"][new_rows, :] = (
        old_sd["gru.weight_ih_l0"]
        .index_select(0, old_rows)
        .index_select(1, old_gru_input_cols)
    )
    new_sd["gru.weight_hh_l0"][new_rows, :] = (
        old_sd["gru.weight_hh_l0"].index_select(0, old_rows).index_select(1, keep_gru_t)
    )
    new_sd["gru.bias_ih_l0"][new_rows] = old_sd["gru.bias_ih_l0"].index_select(
        0, old_rows
    )
    new_sd["gru.bias_hh_l0"][new_rows] = old_sd["gru.bias_hh_l0"].index_select(
        0, old_rows
    )


def _copy_layernorm(old_sd, new_sd, prefix, keep_indices):
    # LayerNorm has only learnable affine parameters (weight, bias);
    # unlike BatchNorm, it has no running_mean/running_var state.
    keep_t = torch.tensor(keep_indices, dtype=torch.long)
    for attr in ["weight", "bias"]:
        new_sd[f"{prefix}.{attr}"][:] = old_sd[f"{prefix}.{attr}"].index_select(
            0, keep_t
        )


def structurally_prune_blstm_gru(model, plan, dev=None):
    dev = dev if dev is not None else next(model.parameters()).device
    model_cpu = copy.deepcopy(model).cpu()
    old_sd = model_cpu.state_dict()

    new_model = HybridBLSTM_GRU(
        input_size=model.input_size,
        num_classes=model.num_classes,
        blstm_hidden=plan.new_lstm_hidden,
        gru_hidden=plan.new_gru_hidden,
        dense_hidden=plan.new_fc_hidden,
        dropout=model.dropout_p,
        flatten_parameters=True,
    ).cpu()
    new_sd = new_model.state_dict()

    _copy_lstm_direction(
        old_sd, new_sd, model.lstm_hidden, plan.new_lstm_hidden, plan.keep_lstm_fwd, ""
    )
    _copy_lstm_direction(
        old_sd,
        new_sd,
        model.lstm_hidden,
        plan.new_lstm_hidden,
        plan.keep_lstm_bwd,
        "_reverse",
    )
    _copy_gru(
        old_sd,
        new_sd,
        model.gru_hidden,
        plan.new_gru_hidden,
        plan.keep_gru,
        model.lstm_hidden,
        plan.keep_lstm_fwd,
        plan.keep_lstm_bwd,
    )

    # Copy LayerNorm layers
    old_lstm_out_cols = list(plan.keep_lstm_fwd) + [
        model.lstm_hidden + i for i in plan.keep_lstm_bwd
    ]
    _copy_layernorm(old_sd, new_sd, "ln_blstm", old_lstm_out_cols)
    _copy_layernorm(old_sd, new_sd, "ln_gru", plan.keep_gru)
    _copy_layernorm(old_sd, new_sd, "ln_fc", plan.keep_fc_hidden)

    keep_gru_t = torch.tensor(plan.keep_gru, dtype=torch.long)
    keep_fc_t = torch.tensor(plan.keep_fc_hidden, dtype=torch.long)
    new_sd["fc1.weight"][:, :] = (
        old_sd["fc1.weight"].index_select(0, keep_fc_t).index_select(1, keep_gru_t)
    )
    new_sd["fc1.bias"][:] = old_sd["fc1.bias"].index_select(0, keep_fc_t)
    new_sd["fc_out.weight"][:, :] = old_sd["fc_out.weight"].index_select(1, keep_fc_t)
    new_sd["fc_out.bias"][:] = old_sd["fc_out.bias"]

    new_model.load_state_dict(new_sd, strict=True)
    return new_model.to(dev)


def zero_shot_depgraph_prune_model(
    model, pruning_rate, include_bias_in_score=True, dev=None
):
    plan = build_zero_shot_depgraph_plan(
        model, pruning_rate=pruning_rate, include_bias_in_score=include_bias_in_score
    )
    pruned_model = structurally_prune_blstm_gru(model, plan, dev=dev)
    return pruned_model, plan


def build_model(
    num_features,
    num_classes,
    device,
    *,
    init_kaiming=True,
    lstm_hidden=None,
    gru_hidden=None,
    fc_hidden=None,
):
    model = HybridBLSTM_GRU(
        input_size=num_features,
        num_classes=num_classes,
        blstm_hidden=BLSTM_HIDDEN if lstm_hidden is None else lstm_hidden,
        gru_hidden=GRU_HIDDEN if gru_hidden is None else gru_hidden,
        dense_hidden=DENSE_HIDDEN if fc_hidden is None else fc_hidden,
        dropout=DROPOUT,
        flatten_parameters=True,
    )
    if init_kaiming:
        model.apply(init_weights_kaiming)
    return model.to(device)


def select_clients_for_round(rng, num_clients, fraction):
    num_selected = max(1, int(num_clients * fraction))
    selected = rng.choice(num_clients, size=num_selected, replace=False)
    return sorted(selected.tolist())


def fedavg_paper_uniform(client_params_list):
    # Lightweight-Fed-NIDS uses FedAvg as described in the paper: averaging all
    # participating client models, i.e. uniform averaging, without weighting
    # by local sample count. The DepGraph paper does not define a separate aggregation method.
    if not client_params_list:
        raise ValueError("No client updates to aggregate.")
    num_updates = len(client_params_list)
    avg_params = OrderedDict()
    for key in client_params_list[0].keys():
        first_tensor = client_params_list[0][key]
        if not torch.is_floating_point(first_tensor):
            avg_params[key] = first_tensor.clone()
            continue
        avg_tensor = torch.zeros_like(first_tensor, dtype=torch.float32)
        for params in client_params_list:
            avg_tensor += (
                params[key].detach().float().to(avg_tensor.device) / num_updates
            )
        avg_params[key] = avg_tensor.to(dtype=first_tensor.dtype)
    return avg_params


def train_one_client(model, loader, *, epochs, lr, dev):
    model.train()
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item())
            num_batches += 1
    mean_loss = running_loss / max(num_batches, 1)
    return get_params(model), mean_loss


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

    # The notebook audits pruning with PyTorch's default initialization,
    # not the optional local Kaiming initializer. Its training cell reseeds
    # after this audit, so the audit never alters the training RNG stream.
    test_full = build_model(num_features, num_classes, device, init_kaiming=False)
    test_pruned, test_plan = zero_shot_depgraph_prune_model(
        test_full,
        PRUNING_RATE,
        INCLUDE_BIAS_IN_SCORE,
        dev=device,
    )
    dummy_in = torch.randn(2, 1, num_features, device=device)
    dummy_out = test_pruned(dummy_in)
    assert dummy_out.shape == (2, num_classes), "Model structure error after pruning!"
    full_p = count_parameters(test_full)
    pruned_p = count_parameters(test_pruned)
    print("===== ZERO-SHOT DEPGRAPH PRUNING AUDIT =====")
    print(
        f"Original model: {full_p:,} parameters (BLSTM={test_full.lstm_hidden}, "
        f"GRU={test_full.gru_hidden}, FC={test_full.fc_hidden})"
    )
    print(
        f"Pruned model: {pruned_p:,} parameters (BLSTM={test_pruned.lstm_hidden}, "
        f"GRU={test_pruned.gru_hidden}, FC={test_pruned.fc_hidden})"
    )
    print(
        f"Actual parameter reduction: {1.0 - pruned_p / full_p:.2%} "
        f"(prune dimension: {PRUNING_RATE:.0%})"
    )

    set_seed(SEED)
    rng = np.random.default_rng(SEED)
    full_global_model = build_model(
        num_features, num_classes, device, init_kaiming=False
    )
    global_model, prune_plan = zero_shot_depgraph_prune_model(
        full_global_model,
        PRUNING_RATE,
        INCLUDE_BIAS_IN_SCORE,
        dev=device,
    )
    metrics_log, payload_events, client_diagnostics = [], [], []

    for rnd in range(1, num_rounds + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        global_params = get_params(global_model)
        download_payload = summarize_payload(global_params)
        selected_clients = select_clients_for_round(rng, num_clients, CLIENT_FRACTION)
        local_t0 = time.perf_counter()
        client_params_list, client_train_losses = [], []
        for cid in selected_clients:
            local_model = build_model(
                num_features,
                num_classes,
                device,
                init_kaiming=False,
                lstm_hidden=prune_plan.new_lstm_hidden,
                gru_hidden=prune_plan.new_gru_hidden,
                fc_hidden=prune_plan.new_fc_hidden,
            )
            set_params(local_model, global_params)
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "downlink",
                    "model",
                    download_payload,
                )
            )
            local_params, client_train_loss = train_one_client(
                local_model,
                client_trainloaders[cid],
                epochs=num_epochs,
                lr=learning_rate,
                dev=device,
            )
            client_params_list.append(local_params)
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "uplink",
                    "update",
                    summarize_payload(local_params),
                )
            )
            client_train_losses.append(client_train_loss)
            client_diagnostics.append(
                {"round": rnd, "client_id": cid, "train_loss": client_train_loss}
            )

        local_time_sec = time.perf_counter() - local_t0
        server_t0 = time.perf_counter()
        averaged_params = fedavg_paper_uniform(client_params_list)
        set_params(global_model, averaged_params)
        eval_res = evaluate(
            global_model, global_testloader, device, num_classes, include_loss=True
        )
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
                    for key, value in eval_res.items()
                    if key not in {"preds", "labels"}
                },
                "train_loss": float(np.mean(client_train_losses)),
                "local_time_sec": round(local_time_sec, 2),
                "server_time_sec": round(server_time_sec, 2),
                "peak_vram_mb": round(float(peak_vram_mb), 2),
                "selected_clients": str(selected_clients),
            }
        )
        ckpt_path = checkpoint_dir / f"round_{rnd:02d}_zeroshot.pt"
        save_model_checkpoint(global_model.state_dict(), ckpt_path)
        print(
            f"Round {rnd:02d}/{num_rounds:02d} | Clients={selected_clients} | "
            f"Loss={eval_res['loss']:.4f} | Acc={eval_res['accuracy']:.4f} | "
            f"F1_M={eval_res['f1_macro']:.4f} | F1_W={eval_res['f1_weighted']:.4f} | "
            f"Local={local_time_sec:.1f}s | Server={server_time_sec:.1f}s"
        )

    return {
        "model": global_model,
        "metrics_log": metrics_log,
        "payload_events": payload_events,
        "prune_plan": prune_plan,
        "model_dimensions": prune_plan.summary(),
        "full_global_model": full_global_model,
        "client_diagnostics": client_diagnostics,
        "pruning_audit": {
            "full_model": test_full,
            "pruned_model": test_pruned,
            "prune_plan": test_plan,
            "full_parameters": full_p,
            "pruned_parameters": pruned_p,
            "parameter_reduction": 1.0 - pruned_p / full_p,
            "output_shape": tuple(dummy_out.shape),
        },
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
