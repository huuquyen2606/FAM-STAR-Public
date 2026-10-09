"""JoPEQ port of Quantization/JoPEQ/{10,20,50} clients notebooks.

The nominal B=4 scalar quantizer emits 17 symbols, packed in five bits.
The deterministic shared/private seed protocol and original privacy assumptions
are retained; only shared model, data, metrics and export plumbing is reused.
"""

from __future__ import annotations

import os

# Must precede torch imports for deterministic CUDA RNN operations.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import io
import math
import time
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHOD_NAME = "JoPEQ"
METHOD_SLUG = "jopeq"
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

JOPEQ_EPSILON = 3.0
JOPEQ_B = 4.0
JOPEQ_GAMMA = 1.5 * math.sqrt(1.0 + 2.0 * (2.0 / JOPEQ_EPSILON) ** 2)
JOPEQ_NON_QUANTIZED_KEYS = {"num_batches_tracked"}
JOPEQ_BASE_SEED = 42


def jopeq_symbol_bits(B):
    if B != int(B) or not 1 <= B <= 15:
        raise ValueError("JoPEQ B must be an integer between 1 and 15")
    # The symmetric quantizer emits -2^(B-1), ..., +2^(B-1):
    # 2^B + 1 symbols require B + 1 fixed-width bits.
    return int(B) + 1


def serialize_jopeq_payload(delta, encoded, zeta, sample_count, B):
    width = jopeq_symbol_bits(B)
    radius = 1 << (int(B) - 1)
    float_keys = [key for key, value in delta.items() if torch.is_floating_point(value)]
    codes = np.concatenate(
        [encoded[key].detach().cpu().numpy().reshape(-1) for key in float_keys]
    )
    if codes.size and (codes.min() < -radius or codes.max() > radius):
        raise ValueError("Quantizer emitted a symbol outside its declared codebook")
    shifts = np.arange(width - 1, -1, -1, dtype=np.int32)
    symbols = codes.astype(np.int32, copy=False) + radius
    packed = np.packbits(((symbols[:, None] >> shifts) & 1).astype(np.uint8).ravel())
    integer_buffers = {
        key: encoded[key].detach().cpu().contiguous()
        for key, value in delta.items()
        if not torch.is_floating_point(value)
    }
    # Model layout is agreed before training. Only scale is per-update metadata;
    # sample counts are local aggregation metadata, not a JoPEQ wire field.
    message = {
        "symbols": torch.from_numpy(packed),
        "zeta": float(zeta),
        "integer_buffers": integer_buffers,
    }
    buffer = io.BytesIO()
    torch.save(message, buffer)
    wire = buffer.getvalue()
    integer_bytes = sum(t.numel() * t.element_size() for t in integer_buffers.values())
    values_bytes = int(packed.nbytes + integer_bytes)
    total_params = int(sum(t.numel() for t in delta.values()))
    nnz = int(sum(torch.count_nonzero(t).item() for t in encoded.values()))
    return wire, {
        "raw_bytes": int(sum(t.numel() * t.element_size() for t in delta.values())),
        "serialized_bytes": len(wire),
        "values_bytes": values_bytes,
        "indices_bytes": 0,
        "scales_bytes": 8,
        "mask_bytes": 0,
        "prototype_bytes": 0,
        "metadata_bytes": len(wire) - values_bytes - 8,
        "symbol_bits": width,
        "nnz": nnz,
        "total_params": total_params,
        "sparsity": float(1.0 - nnz / max(total_params, 1)),
    }


def deserialize_jopeq_payload(wire, template, B):
    width = jopeq_symbol_bits(B)
    radius = 1 << (int(B) - 1)
    message = torch.load(io.BytesIO(wire), map_location="cpu", weights_only=True)
    count = sum(t.numel() for t in template.values() if torch.is_floating_point(t))
    packed = message["symbols"].numpy()
    if packed.nbytes != (count * width + 7) // 8:
        raise ValueError("Incorrect packed JoPEQ update length")
    shifts = np.arange(width - 1, -1, -1, dtype=np.int32)
    codes = (
        np.unpackbits(packed, count=count * width).reshape(-1, width) @ (1 << shifts)
    ).astype(np.int32) - radius
    if codes.size and (codes.min() < -radius or codes.max() > radius):
        raise ValueError("Invalid packed JoPEQ symbol")
    integer_buffers = message["integer_buffers"]
    if set(integer_buffers) != {
        key for key, value in template.items() if not torch.is_floating_point(value)
    }:
        raise ValueError("JoPEQ integer buffers do not match the model")
    encoded = {}
    start = 0
    for key, reference in template.items():
        if torch.is_floating_point(reference):
            end = start + reference.numel()
            encoded[key] = (
                torch.from_numpy(codes[start:end].copy())
                .reshape(reference.shape)
                .to(reference.device)
            )
            start = end
        else:
            value = integer_buffers[key]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"Invalid JoPEQ buffer {key}")
            encoded[key] = value.to(reference.device)
    return encoded, message["zeta"]


def payload_error_norm(reference, reconstructed):
    squared = 0.0
    for key in reference:
        if torch.is_tensor(reference[key]):
            diff = reference[key].float() - reconstructed[key].float()
            squared += float(torch.sum(diff * diff).item())
    return float(math.sqrt(squared))


def local_train(model, loader, epochs, lr, device):
    model.train()
    optimizer = optim.Adamax(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item())
            num_batches += 1
    mean_loss = running_loss / max(num_batches, 1)
    return get_params(model), mean_loss


def _jopeq_scale(delta):
    # Scale tren toan bo cac update floating-point duoc truyen di,
    # bao gom ca running_mean/running_var; chi bo qua buffer integer.
    flat = torch.cat(
        [v.flatten().float() for v in delta.values() if torch.is_floating_point(v)]
    )
    M = flat.numel()
    norm = flat.norm()
    zeta = 1.0 / ((3.0 / math.sqrt(M)) * norm + 1e-12)
    return float(zeta), M


def jopeq_encode(delta, epsilon, shared_seed, private_seed, B=None):
    if B is None:
        B = JOPEQ_B
    # [Phia Client] JoPEQ Encoding:
    #   1. Scale delta voi he so zeta
    #   2. Them PPN scalar da dieu chinh de bu tru phuong sai dither
    #   3. Them dither (tu shared_seed) theo Subtractive Dithered Quantization (SDQ)
    #   4. Luong tu hoa scalar (Lattice quantization voi L=1)
    zeta, M = _jopeq_scale(delta)
    gamma = JOPEQ_GAMMA
    step = (2.0 * gamma) / ((2**B) + 1)
    edge = gamma - (step / 2.0)

    # Sensitivity sau khi scale: range [-1,1] => GS = 2
    # Scale Laplace muc tieu cua tong distortion sau SDQ
    global_sensitivity = 2.0
    target_lap_scale = global_sensitivity / epsilon
    target_variance = 2.0 * (target_lap_scale**2)
    dither_variance = (step**2) / 12.0
    ppn_variance = max(target_variance - dither_variance, 0.0)
    lap_scale = math.sqrt(ppn_variance / 2.0)

    # RNG rieng tu doc lap voi shared seed — khong gui seed nay cho server
    rng_priv = torch.Generator()
    rng_priv.manual_seed(private_seed % (2**32))
    # RNG chung (shared) de sinh dither — server TAO LAI duoc chinh xac
    rng_shared = torch.Generator()
    rng_shared.manual_seed(shared_seed % (2**32))

    encoded = {}
    for key, val in delta.items():
        # Quantize va bao ve moi update floating-point; buffer integer duoc FedAvg xu ly.
        if torch.is_floating_point(val):
            v = val.float().flatten()
            device_v = v.device

            # Buoc 1: Scale
            v_scaled = zeta * v

            # Buoc 2: Them PPN Laplace da tru dither variance (inverse CDF)
            u = (
                torch.rand(v.shape, generator=rng_priv, device="cpu").to(device_v) - 0.5
            )  # uniform(-0.5, 0.5)
            laplace_noise = -lap_scale * u.sign() * torch.log1p(-2.0 * u.abs())
            v_noisy = v_scaled + laplace_noise

            # Buoc 3: Them dither uniform(-step/2, step/2)
            dither = (
                torch.rand(v.shape, generator=rng_shared, device="cpu").to(device_v)
                - 0.5
            ) * step

            # Buoc 4: Scalar lattice quantization tren support [-edge, edge]
            v_dithered = v_noisy + dither
            v_q = torch.round(v_dithered / step) * step
            v_q = torch.clamp(v_q, -edge, edge)

            # Luu ma so roi rac; server khoi phuc lai gia tri q*step.
            encoded[key] = torch.round(v_q / step).to(torch.int32).reshape(val.shape)
        else:
            # Buffer integer khong phai update continuous de quantize.
            encoded[key] = val.clone()

    return encoded, zeta


def jopeq_decode(encoded, zeta, shared_seed, B=None):
    if B is None:
        B = JOPEQ_B
    # [Phia Server] JoPEQ Decoding:
    #   1. Tru dither (subtractive dithering) — tai tao chinh xac tu shared_seed
    #   2. Scale nguoc voi 1/zeta
    step = (2.0 * JOPEQ_GAMMA) / ((2**B) + 1)
    rng_shared = torch.Generator()
    rng_shared.manual_seed(shared_seed % (2**32))

    decoded = {}
    for key, val in encoded.items():
        if key not in JOPEQ_NON_QUANTIZED_KEYS:
            v_q = val.float().flatten() * step
            device_v = v_q.device
            dither = (
                torch.rand(v_q.shape, generator=rng_shared, device="cpu").to(device_v)
                - 0.5
            ) * step
            # Tru dither roi scale nguoc de phuc hoi uoc tinh delta
            decoded[key] = ((v_q - dither) / (zeta + 1e-12)).reshape(val.shape)
        else:
            # Buffer integer duoc giu nguyen de FedAvg xu ly.
            decoded[key] = val.clone()

    return decoded


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
    """Train, encode actual messages, decode, and aggregate JoPEQ deltas."""
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
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
    protocol_state = {}
    print(
        f"Starting JoPEQ: {num_rounds} rounds | {num_clients} clients | "
        f"batch={batch_size} | epsilon={JOPEQ_EPSILON} | nominal B={JOPEQ_B}; "
        f"packed={jopeq_symbol_bits(JOPEQ_B)} bits/float"
    )

    for rnd in range(1, num_rounds + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        global_params = get_params(global_model)
        # Use the same measured bytes for the actual client decode.
        download_wire, download_payload = summarize_payload(
            global_params, return_wire=True
        )
        download_payload = {**download_payload, "symbol_bits": 0}

        local_t0 = time.perf_counter()
        upload_wires = []
        upload_summaries = []
        raw_delta_list = []
        seeds_list = []
        client_sizes = []
        client_train_losses = []
        for cid in range(num_clients):
            local_model = HybridBLSTM_GRU(
                input_size=num_features,
                num_classes=num_classes,
                blstm_hidden=BLSTM_HIDDEN,
                gru_hidden=GRU_HIDDEN,
                dense_hidden=DENSE_HIDDEN,
                dropout=DROPOUT,
            ).to(device)
            set_params(
                local_model,
                torch.load(
                    io.BytesIO(download_wire),
                    map_location=device,
                    weights_only=True,
                ),
            )
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "downlink",
                    "model",
                    download_payload,
                )
            )
            local_params, client_train_loss = local_train(
                local_model,
                client_trainloaders[cid],
                num_epochs,
                learning_rate,
                device,
            )
            delta = {}
            for key in local_params:
                delta[key] = local_params[key] - global_params[key]
            shared_seed = JOPEQ_BASE_SEED * 10000 + rnd * 100 + cid
            private_seed = JOPEQ_BASE_SEED * 1000000 + rnd * 100 + cid
            encoded, zeta = jopeq_encode(
                delta,
                epsilon=JOPEQ_EPSILON,
                shared_seed=shared_seed,
                private_seed=private_seed,
                B=JOPEQ_B,
            )
            upload_wire, upload_summary = serialize_jopeq_payload(
                delta,
                encoded,
                zeta,
                len(client_y_list[cid]),
                JOPEQ_B,
            )
            raw_delta_list.append(delta)
            upload_wires.append(upload_wire)
            upload_summaries.append(upload_summary)
            seeds_list.append(shared_seed)
            client_train_losses.append(client_train_loss)
        local_time_sec = time.perf_counter() - local_t0

        server_t0 = time.perf_counter()
        decoded_list = []
        for cid, (wire, seed) in enumerate(zip(upload_wires, seeds_list)):
            encoded, zeta = deserialize_jopeq_payload(wire, global_params, JOPEQ_B)
            decoded_delta = jopeq_decode(encoded, zeta, seed, B=JOPEQ_B)
            client_sizes.append(len(client_y_list[cid]))
            decoded_list.append(decoded_delta)
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "uplink",
                    "quantized_update",
                    upload_summaries[cid],
                    error_norm=payload_error_norm(raw_delta_list[cid], decoded_delta),
                )
            )
        averaged_delta = fed_avg(decoded_list, client_sizes)
        # Preserve the notebook's float32 addition and integer rounding order.
        new_global_params = {}
        for key in global_params:
            if global_params[key].dtype in [
                torch.int,
                torch.long,
                torch.int32,
                torch.int64,
            ]:
                new_global_params[key] = (
                    (global_params[key].float() + averaged_delta[key].float())
                    .round()
                    .to(global_params[key].dtype)
                )
            else:
                new_global_params[key] = (
                    global_params[key].float() + averaged_delta[key].float()
                )
        set_params(global_model, new_global_params)
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
                    if key not in {"preds", "labels"}
                },
                "train_loss": float(np.mean(client_train_losses)),
                "local_time_sec": round(local_time_sec, 2),
                "server_time_sec": round(server_time_sec, 2),
                "peak_vram_mb": round(float(peak_vram_mb), 2),
            }
        )
        ckpt_path = checkpoint_dir / f"round_{rnd:02d}_jopeq.pt"
        save_model_checkpoint(global_model.state_dict(), ckpt_path)
        print(
            f"Round {rnd:02d}/{num_rounds} | Acc={metrics['accuracy']:.4f} | "
            f"F1-M={metrics['f1_macro']:.4f} | F1-W={metrics['f1_weighted']:.4f} | "
            f"Local={local_time_sec:.1f}s | Server={server_time_sec:.1f}s"
        )
        if rnd == num_rounds:
            protocol_state = {
                "round": rnd,
                "download_wire": download_wire,
                "upload_wires": upload_wires,
                "upload_summaries": upload_summaries,
                "shared_seeds": seeds_list,
                "client_sizes": client_sizes,
                "client_train_losses": client_train_losses,
                "raw_deltas": raw_delta_list,
                "decoded_deltas": decoded_list,
                "averaged_delta": averaged_delta,
            }
    return {
        "model": global_model,
        "metrics_log": metrics_log,
        "payload_events": payload_events,
        "protocol_state": protocol_state,
        "jopeq_epsilon": JOPEQ_EPSILON,
        "jopeq_B": JOPEQ_B,
        "jopeq_gamma": JOPEQ_GAMMA,
        "symbol_bits": jopeq_symbol_bits(JOPEQ_B),
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
        extra_tables=result.get("extra_tables"),
    )
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
