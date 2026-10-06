"""Reliable local/global prototypes used by FAM-STAR."""

from __future__ import annotations

import math

import numpy as np
import torch

from .compression import (
    pack_fp32_scale,
    quantize_int8,
    serialize_payload,
)


def _extract_embeddings(model, xb):
    _, embeddings = model(xb, return_embedding=True)
    return embeddings


def compute_local_prototypes(
    model,
    loader,
    device,
    num_classes: int,
):
    """Compute local class prototypes, supports, and intra-class dispersion."""
    was_training = model.training
    model.eval()

    sums = None
    squared_norm_sums = None
    counts = torch.zeros(
        num_classes,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(
                device,
                non_blocking=True,
            ).long()
            embeddings = _extract_embeddings(model, xb)

            if sums is None:
                sums = torch.zeros(
                    num_classes,
                    embeddings.shape[1],
                    dtype=torch.float32,
                    device=device,
                )
                squared_norm_sums = torch.zeros(
                    num_classes,
                    dtype=torch.float32,
                    device=device,
                )

            sums.index_add_(0, yb, embeddings)
            squared_norm_sums.index_add_(
                0,
                yb,
                embeddings.pow(2).sum(dim=1),
            )
            counts.index_add_(
                0,
                yb,
                torch.ones_like(yb, dtype=torch.long),
            )

    model.train(was_training)

    if sums is None:
        return {}, {}, {}

    prototypes = {}
    supports = {}
    dispersions = {}

    for class_id in range(num_classes):
        count = int(counts[class_id].item())
        if count <= 0:
            continue

        prototype = sums[class_id] / float(count)
        mean_squared_norm = (
            squared_norm_sums[class_id] / float(count)
        )
        dispersion = torch.clamp(
            mean_squared_norm - prototype.pow(2).sum(),
            min=0.0,
        )

        prototypes[class_id] = prototype.detach().cpu()
        supports[class_id] = count
        dispersions[class_id] = float(dispersion.item())

    return prototypes, supports, dispersions


def quantize_prototype_package(
    prototypes,
    supports,
    dispersions,
    eps: float = 1e-8,
    scale_bytes: int = 4,
):
    """INT8 prototype values + support/dispersion reliability metadata."""
    class_payload = {}
    dequantized = {}
    raw_bytes = 0
    prototype_bytes = 0
    scales_bytes = 0

    for class_id in sorted(prototypes):
        prototype = (
            prototypes[class_id]
            .detach()
            .cpu()
            .to(torch.float32)
            .reshape(-1)
        )
        quantized, scale = quantize_int8(
            prototype,
            eps=eps,
        )
        dequantized[class_id] = (
            quantized.to(torch.float32) * scale
        )
        support = int(supports.get(class_id, 0))
        dispersion = max(
            float(dispersions.get(class_id, 0.0)),
            0.0,
        )

        class_payload[int(class_id)] = {
            "values_q": quantized,
            "scale": pack_fp32_scale(scale),
            "support": support,
            "dispersion": dispersion,
            "shape": tuple(prototype.shape),
        }

        raw_bytes += int(prototype.numel() * 4)
        prototype_bytes += int(quantized.numel())
        scales_bytes += int(scale_bytes)

    payload = {
        "__prototypes__": {
            "kind": "reliable_prototypes_int8",
            "classes": class_payload,
        }
    }
    serialized_bytes = serialize_payload(payload)
    metadata_bytes = max(
        serialized_bytes - prototype_bytes - scales_bytes,
        0,
    )

    summary = {
        "raw_bytes": raw_bytes,
        "serialized_bytes": serialized_bytes,
        "values_bytes": 0,
        "indices_bytes": 0,
        "scales_bytes": scales_bytes,
        "mask_bytes": 0,
        "prototype_bytes": prototype_bytes,
        "metadata_bytes": metadata_bytes,
        "nnz": 0,
        "total_params": int(
            sum(
                int(prototypes[class_id].numel())
                for class_id in prototypes
            )
        ),
        "sparsity": 0.0,
    }

    return {
        "prototypes": dequantized,
        "supports": {
            int(key): int(value)
            for key, value in supports.items()
        },
        "dispersions": {
            int(key): float(value)
            for key, value in dispersions.items()
        },
        "payload": payload,
        "summary": summary,
    }


def aggregate_prototypes(
    client_packages,
    previous_prototypes=None,
    eps: float = 1e-8,
):
    """Reliability fusion: log(1+n)/(dispersion+eps), class by class."""
    previous_prototypes = previous_prototypes or {}
    weighted_sums = {}
    weight_sums = {}
    reliability_values = {}
    contributing_clients = set()

    for client_id, package in enumerate(client_packages):
        prototypes = package.get("prototypes", {})
        supports = package.get("supports", {})
        dispersions = package.get("dispersions", {})

        if prototypes:
            contributing_clients.add(client_id)

        for class_id, prototype in prototypes.items():
            class_id = int(class_id)
            support = int(supports.get(class_id, 0))
            if support <= 0:
                continue

            dispersion = max(
                float(dispersions.get(class_id, 0.0)),
                0.0,
            )
            reliability = math.log1p(support) / (
                dispersion + float(eps)
            )

            weighted_sums[class_id] = (
                weighted_sums.get(
                    class_id,
                    torch.zeros_like(prototype),
                )
                + prototype.to(torch.float32) * reliability
            )
            weight_sums[class_id] = (
                weight_sums.get(class_id, 0.0)
                + reliability
            )
            reliability_values.setdefault(
                class_id,
                [],
            ).append(reliability)

    global_prototypes = {}
    for class_id in sorted(
        set(weighted_sums) | set(previous_prototypes)
    ):
        if (
            class_id in weight_sums
            and weight_sums[class_id] > 0.0
        ):
            global_prototypes[class_id] = (
                weighted_sums[class_id]
                / weight_sums[class_id]
            ).detach().cpu()
        elif class_id in previous_prototypes:
            global_prototypes[class_id] = (
                previous_prototypes[class_id]
                .detach()
                .cpu()
                .clone()
            )

    all_reliabilities = [
        value
        for values in reliability_values.values()
        for value in values
    ]
    per_class_reliability = {
        int(class_id): float(np.mean(values))
        for class_id, values in reliability_values.items()
    }

    stats = {
        "num_classes": int(len(global_prototypes)),
        "contributing_clients": int(
            len(contributing_clients)
        ),
        "mean_reliability": float(
            np.mean(all_reliabilities)
            if all_reliabilities
            else 0.0
        ),
        "per_class_reliability": per_class_reliability,
    }
    return global_prototypes, stats
