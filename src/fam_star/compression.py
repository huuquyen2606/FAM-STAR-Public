"""Compact FAM-STAR uplink/downlink serialization and accounting."""

from __future__ import annotations

import copy
import io
import math
import struct

import numpy as np
import torch

from .topology import is_prunable_parameter


def serialize_payload(payload) -> int:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return int(buffer.getbuffer().nbytes)


def pack_fp32_scale(scale: float) -> bytes:
    """Encode one per-tensor scale as exactly four FP32 wire bytes."""
    return struct.pack("<f", float(scale))


def pack_bool_mask(mask, bitorder: str = "little"):
    """Losslessly pack a boolean mask at one bit per position."""
    if mask is None:
        return None
    flat = (
        mask.detach()
        .cpu()
        .reshape(-1)
        .bool()
        .numpy()
        .astype(np.uint8, copy=False)
    )
    packed = np.packbits(flat, bitorder=bitorder)
    return torch.from_numpy(
        packed.copy()
    ).to(torch.uint8).contiguous()


def unpack_bool_mask(
    packed,
    numel: int,
    shape=None,
    bitorder: str = "little",
):
    if packed is None:
        return None

    raw = packed.detach().cpu().to(torch.uint8).numpy()
    unpacked = np.unpackbits(
        raw,
        count=int(numel),
        bitorder=bitorder,
    ).astype(np.bool_)

    tensor = torch.from_numpy(unpacked.copy())
    if shape is not None:
        tensor = tensor.reshape(tuple(shape))
    return tensor


def quantize_int8(values, eps: float = 1e-8):
    values = (
        values.detach()
        .cpu()
        .to(torch.float32)
        .reshape(-1)
        .contiguous()
    )
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.int8), float(eps)

    max_abs = float(values.abs().max().item())
    scale = max(max_abs / 127.0, float(eps))
    quantized = torch.clamp(
        torch.round(values / scale),
        min=-127,
        max=127,
    ).to(torch.int8)
    return quantized, float(scale)


def compress_sparse_delta(
    local_state,
    global_state,
    masks,
    proposal_masks=None,
    previous_residuals=None,
    eps: float = 1e-8,
    min_dim: int = 2,
    bitorder: str = "little",
    scale_bytes: int = 4,
):
    """INT8 sparse delta + error feedback + bit-packed support/topology."""
    previous_residuals = previous_residuals or {}
    payload = {}
    dequantized_delta = {}
    next_residuals = {}

    total_params = 0
    nnz = 0
    raw_bytes = 0
    values_bytes = 0
    indices_bytes = 0
    scales_bytes = 0
    mask_bytes = 0
    support_bitmap_bytes = 0
    proposal_bitmap_bytes = 0
    error_sq = 0.0

    for name, local_tensor in local_state.items():
        if not torch.is_tensor(local_tensor):
            continue

        local_cpu = local_tensor.detach().cpu().contiguous()
        global_cpu = (
            global_state[name].detach().cpu().contiguous()
        )

        total_params += int(local_cpu.numel())
        raw_bytes += int(local_cpu.numel() * 4)

        if not torch.is_floating_point(local_cpu):
            payload[name] = {
                "kind": "unchanged",
                "shape": tuple(local_cpu.shape),
            }
            dequantized_delta[name] = torch.zeros_like(local_cpu)
            continue

        delta = (
            local_cpu.to(torch.float32)
            - global_cpu.to(torch.float32)
        )

        mask = masks.get(name) if masks is not None else None
        prunable = (
            mask is not None
            and is_prunable_parameter(name, local_cpu, min_dim=min_dim)
        )

        proposal_mask_bits = None
        if prunable and proposal_masks is not None:
            if name not in proposal_masks:
                raise KeyError(
                    f"Missing topology proposal for {name}"
                )
            proposal_mask_bits = pack_bool_mask(
                proposal_masks[name],
                bitorder=bitorder,
            )

        if prunable:
            mask_cpu = (
                mask.detach()
                .cpu()
                .reshape_as(delta)
                .to(torch.float32)
            )
            delta = delta * mask_cpu

        previous = previous_residuals.get(name)
        if (
            previous is None
            or tuple(previous.shape) != tuple(delta.shape)
        ):
            previous = torch.zeros_like(delta)
        else:
            previous = (
                previous.detach()
                .cpu()
                .to(torch.float32)
                .reshape_as(delta)
            )

        u = (delta + previous).contiguous()
        flat_u = u.reshape(-1)

        if prunable:
            support_bool = flat_u != 0
            support_bits = pack_bool_mask(
                support_bool,
                bitorder=bitorder,
            )
            values = flat_u[support_bool].contiguous()
            quantized, scale = quantize_int8(values, eps=eps)

            dequantized_flat = torch.zeros_like(flat_u)
            if quantized.numel() > 0:
                dequantized_flat[support_bool] = (
                    quantized.to(torch.float32) * scale
                )
            dequantized = dequantized_flat.reshape_as(u)

            payload[name] = {
                "kind": "sparse_int8_delta_bitpacked_support",
                "support_bits": support_bits,
                "support_numel": int(flat_u.numel()),
                "values_q": quantized,
                "scale": pack_fp32_scale(scale),
                "shape": tuple(u.shape),
            }

            support_nbytes = int(
                support_bits.numel() * support_bits.element_size()
            )
            support_bitmap_bytes += support_nbytes
            mask_bytes += support_nbytes

            if proposal_mask_bits is not None:
                payload[name]["next_mask_bits"] = proposal_mask_bits
                payload[name]["next_mask_numel"] = int(
                    proposal_masks[name].numel()
                )
                proposal_nbytes = int(
                    proposal_mask_bits.numel()
                    * proposal_mask_bits.element_size()
                )
                proposal_bitmap_bytes += proposal_nbytes
                mask_bytes += proposal_nbytes

            nnz += int(quantized.numel())

        else:
            quantized, scale = quantize_int8(flat_u, eps=eps)
            dequantized = (
                quantized.to(torch.float32) * scale
            ).reshape_as(u)
            payload[name] = {
                "kind": "dense_int8_delta",
                "values_q": quantized.reshape_as(u),
                "scale": pack_fp32_scale(scale),
                "shape": tuple(u.shape),
            }
            nnz += int(quantized.numel())

        values_bytes += int(quantized.numel())
        scales_bytes += int(scale_bytes)

        residual = (u - dequantized).contiguous()
        next_residuals[name] = residual
        dequantized_delta[name] = dequantized

        error_sq += float(residual.pow(2).sum().item())

    serialized_bytes = serialize_payload(payload)
    metadata_bytes = max(
        serialized_bytes
        - values_bytes
        - indices_bytes
        - scales_bytes
        - mask_bytes,
        0,
    )
    error_norm = math.sqrt(max(error_sq, 0.0))

    summary = {
        "raw_bytes": raw_bytes,
        "serialized_bytes": serialized_bytes,
        "values_bytes": values_bytes,
        "indices_bytes": indices_bytes,
        "scales_bytes": scales_bytes,
        "mask_bytes": mask_bytes,
        "support_bitmap_bytes": support_bitmap_bytes,
        "proposal_bitmap_bytes": proposal_bitmap_bytes,
        "broadcast_mask_bitmap_bytes": 0,
        "prototype_bytes": 0,
        "metadata_bytes": metadata_bytes,
        "nnz": nnz,
        "total_params": total_params,
        "sparsity": float(
            1.0 - nnz / max(total_params, 1)
        ),
    }

    return {
        "dequantized_delta": dequantized_delta,
        "residuals": next_residuals,
        "summary": summary,
        "error_norm": error_norm,
        "payload": payload,
    }


def summarize_sparse_model_payload(
    state_dict,
    masks,
    global_prototypes=None,
    min_dim: int = 2,
    bitorder: str = "little",
):
    """Account for sparse FP32 downlink plus optional global prototypes."""
    cpu_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }

    total_params = int(
        sum(tensor.numel() for tensor in cpu_state.values())
    )

    nnz = 0
    values_bytes = 0
    indices_bytes = 0
    mask_bytes = 0
    broadcast_mask_bitmap_bytes = 0
    scales_bytes = 0
    prototype_raw_bytes = 0
    prototype_bytes = 0
    payload = {}

    for name, tensor in cpu_state.items():
        mask = masks.get(name) if masks is not None else None

        if (
            mask is not None
            and is_prunable_parameter(name, tensor, min_dim=min_dim)
        ):
            active_bool = (
                mask.detach().cpu().reshape(-1).bool()
            )
            mask_bits = pack_bool_mask(
                active_bool,
                bitorder=bitorder,
            )
            values = (
                tensor.reshape(-1)[active_bool].contiguous()
            )

            payload[name] = {
                "kind": "sparse_fp32_model_bitpacked_mask",
                "mask_bits": mask_bits,
                "mask_numel": int(active_bool.numel()),
                "values": values,
                "shape": tuple(tensor.shape),
            }

            nnz += int(active_bool.sum().item())
            values_bytes += int(
                values.numel() * values.element_size()
            )
            bit_bytes = int(
                mask_bits.numel() * mask_bits.element_size()
            )
            mask_bytes += bit_bytes
            broadcast_mask_bitmap_bytes += bit_bytes

        else:
            payload[name] = {
                "kind": "dense_fp32_model",
                "values": tensor,
                "shape": tuple(tensor.shape),
            }
            nnz += int(tensor.numel())
            values_bytes += int(
                tensor.numel() * tensor.element_size()
            )

    if global_prototypes:
        fp32_prototypes = {
            int(class_id): (
                prototype.detach()
                .cpu()
                .to(torch.float32)
                .contiguous()
            )
            for class_id, prototype in global_prototypes.items()
        }
        payload["__prototypes__"] = {
            "kind": "dense_fp32_global_prototypes",
            "prototypes": fp32_prototypes,
        }
        prototype_bytes = int(
            sum(
                prototype.numel() * prototype.element_size()
                for prototype in fp32_prototypes.values()
            )
        )
        prototype_raw_bytes = prototype_bytes

    serialized_bytes = serialize_payload(payload)
    metadata_bytes = max(
        serialized_bytes
        - values_bytes
        - indices_bytes
        - scales_bytes
        - mask_bytes
        - prototype_bytes,
        0,
    )

    return {
        "raw_bytes": int(total_params * 4 + prototype_raw_bytes),
        "serialized_bytes": serialized_bytes,
        "values_bytes": values_bytes,
        "indices_bytes": indices_bytes,
        "scales_bytes": scales_bytes,
        "mask_bytes": mask_bytes,
        "support_bitmap_bytes": 0,
        "proposal_bitmap_bytes": 0,
        "broadcast_mask_bitmap_bytes": broadcast_mask_bitmap_bytes,
        "prototype_bytes": prototype_bytes,
        "metadata_bytes": metadata_bytes,
        "nnz": nnz,
        "total_params": total_params,
        "sparsity": float(
            1.0 - nnz / max(total_params, 1)
        ),
    }


def combine_delta_and_prototype_payload(
    delta_result,
    prototype_result,
):
    """Build one FAM-STAR uplink package and account serialized bytes."""
    payload = copy.deepcopy(delta_result["payload"])
    payload["__prototypes__"] = copy.deepcopy(
        prototype_result["payload"]["__prototypes__"]
    )

    serialized_bytes = serialize_payload(payload)
    delta_summary = delta_result["summary"]
    prototype_summary = prototype_result["summary"]
    summary = dict(delta_summary)

    summary["raw_bytes"] = (
        int(delta_summary["raw_bytes"])
        + int(prototype_summary["raw_bytes"])
    )
    summary["serialized_bytes"] = int(serialized_bytes)
    summary["scales_bytes"] = (
        int(delta_summary["scales_bytes"])
        + int(prototype_summary["scales_bytes"])
    )
    summary["prototype_bytes"] = int(
        prototype_summary["prototype_bytes"]
    )
    summary["metadata_bytes"] = max(
        int(serialized_bytes)
        - int(summary["values_bytes"])
        - int(summary["indices_bytes"])
        - int(summary["scales_bytes"])
        - int(summary["mask_bytes"])
        - int(summary["prototype_bytes"]),
        0,
    )
    return payload, summary


def build_payload_event(
    round_id,
    client_id,
    direction,
    payload_type,
    summary,
    error_norm: float = 0.0,
):
    return {
        "round": round_id,
        "client_id": client_id,
        "direction": direction,
        "payload_type": payload_type,
        **summary,
        "error_norm": float(error_norm),
    }
