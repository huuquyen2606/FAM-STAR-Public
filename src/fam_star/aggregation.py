"""FAM-STAR role-specific aggregation (paper Eqs. 5 and 6)."""

from __future__ import annotations

import numpy as np
import torch


def compute_shared_aggregation_weights(
    class_counts,
    sample_weights,
    gamma: float = 0.50,
    eps: float = 1e-12,
):
    """Eq. (5): sample mass blended with diversity-adjusted class evidence."""
    counts = np.asarray(class_counts, dtype=np.float64)
    sw = np.asarray(sample_weights, dtype=np.float64)
    sw = sw / max(float(sw.sum()), eps)

    client_totals = counts.sum(axis=1)
    nonempty = client_totals > 0
    local_distribution = np.zeros_like(counts, dtype=np.float64)
    local_distribution[nonempty] = (
        counts[nonempty] / client_totals[nonempty, None]
    )

    diversity = np.zeros(counts.shape[0], dtype=np.float64)
    simpson = np.sum(local_distribution ** 2, axis=1)
    diversity[nonempty] = 1.0 / np.maximum(simpson[nonempty], eps)

    present = counts.sum(axis=0) > 0
    class_mass = np.zeros_like(counts, dtype=np.float64)
    for class_id in range(counts.shape[1]):
        if not present[class_id]:
            continue
        weighted_support = counts[:, class_id] * diversity
        denom = float(weighted_support.sum())
        if denom <= eps:
            raise ValueError(f"Zero class evidence for class {class_id}")
        class_mass[:, class_id] = weighted_support / denom

    present_ids = np.flatnonzero(present)
    if present_ids.size == 0:
        raise ValueError("No globally present classes")

    diversity_weights = class_mass[:, present_ids].mean(axis=1)
    diversity_weights = (
        diversity_weights / max(float(diversity_weights.sum()), eps)
    )

    shared_weights = (
        (1.0 - float(gamma)) * sw
        + float(gamma) * diversity_weights
    )
    return shared_weights / max(float(shared_weights.sum()), eps)


def compute_head_aggregation_weights(
    class_counts,
    global_class_counts,
    sample_weights,
    gamma: float = 0.50,
    eps: float = 1e-12,
):
    """Eq. (6): per-class classifier rows use class-specific support."""
    counts = np.asarray(class_counts, dtype=np.float64)
    global_counts = np.asarray(global_class_counts, dtype=np.float64)
    sw = np.asarray(sample_weights, dtype=np.float64)
    sw = sw / max(float(sw.sum()), eps)

    num_clients, num_classes = counts.shape
    present = global_counts > 0
    within_class_mass = np.zeros_like(counts, dtype=np.float64)
    within_class_mass[:, present] = (
        counts[:, present] / np.maximum(global_counts[present], eps)
    )

    head_weights = np.zeros(
        (num_clients, num_classes),
        dtype=np.float64,
    )
    for class_id in range(num_classes):
        if present[class_id]:
            head_weights[:, class_id] = (
                (1.0 - float(gamma)) * sw
                + float(gamma) * within_class_mass[:, class_id]
            )
        else:
            head_weights[:, class_id] = sw
        head_weights[:, class_id] /= max(
            float(head_weights[:, class_id].sum()),
            eps,
        )
    return head_weights


def apply_role_specific_aggregation(
    global_params,
    client_delta_list,
    shared_client_weights,
    head_client_weights,
    head_weight_name: str = "fc_out.weight",
    head_bias_name: str = "fc_out.bias",
    eps: float = 1e-12,
):
    """Fuse shared tensors and classifier rows using their FAM-STAR weights."""
    if not client_delta_list:
        raise ValueError("client_delta_list must not be empty")

    num_clients = len(client_delta_list)
    shared = np.asarray(shared_client_weights, dtype=np.float64)
    shared = shared / max(float(shared.sum()), eps)
    head = np.asarray(head_client_weights, dtype=np.float64)

    if shared.shape[0] != num_clients or head.shape[0] != num_clients:
        raise ValueError("Role-specific aggregation client dimension mismatch")

    updated = {}
    for key, base in global_params.items():
        if not torch.is_floating_point(base):
            updated[key] = base.clone()
            continue

        first = client_delta_list[0][key]
        if key == head_weight_name:
            if first.dim() != 2 or head.shape[1] != int(first.shape[0]):
                raise ValueError("fc_out.weight shape mismatch")
            aggregate = torch.zeros_like(first)
            for class_id in range(int(first.shape[0])):
                row = torch.zeros_like(first[class_id])
                for client_id, delta in enumerate(client_delta_list):
                    row.add_(
                        delta[key][class_id].to(dtype=row.dtype),
                        alpha=float(head[client_id, class_id]),
                    )
                aggregate[class_id] = row

        elif key == head_bias_name:
            if first.dim() != 1 or head.shape[1] != int(first.shape[0]):
                raise ValueError("fc_out.bias shape mismatch")
            aggregate = torch.zeros_like(first)
            for class_id in range(int(first.shape[0])):
                value = torch.zeros_like(first[class_id])
                for client_id, delta in enumerate(client_delta_list):
                    value.add_(
                        delta[key][class_id].to(dtype=value.dtype),
                        alpha=float(head[client_id, class_id]),
                    )
                aggregate[class_id] = value

        else:
            aggregate = torch.zeros_like(first)
            for client_id, delta in enumerate(client_delta_list):
                aggregate.add_(
                    delta[key].to(dtype=aggregate.dtype),
                    alpha=float(shared[client_id]),
                )

        updated[key] = base + aggregate.to(
            device=base.device,
            dtype=base.dtype,
        )

    stats = {
        "shared_weight_sum": float(shared.sum()),
        "head_weight_max_sum_error": float(
            np.max(np.abs(head.sum(axis=0) - 1.0))
        ),
    }
    return updated, stats
