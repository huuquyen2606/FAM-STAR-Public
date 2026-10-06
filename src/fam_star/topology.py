"""Dynamic sparse topology used by FAM-STAR."""

from __future__ import annotations

import torch


def is_prunable_parameter(
    name: str,
    tensor: torch.Tensor,
    min_dim: int = 2,
) -> bool:
    # FAM-STAR keeps the classification head dense.
    return tensor.dim() >= min_dim and "fc_out" not in name


def init_sparse_masks(
    model,
    target_sparsity: float = 0.50,
    min_dim: int = 2,
):
    masks = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if is_prunable_parameter(name, param, min_dim=min_dim):
                flat_abs = param.detach().abs().reshape(-1)
                keep = max(
                    1,
                    int(round(flat_abs.numel() * (1.0 - target_sparsity))),
                )
                keep = min(keep, flat_abs.numel())
                flat_mask = torch.zeros_like(flat_abs)
                top_idx = torch.topk(
                    flat_abs,
                    k=keep,
                    largest=True,
                    sorted=False,
                ).indices
                flat_mask[top_idx] = 1.0
                masks[name] = flat_mask.reshape_as(param).clone()
            else:
                masks[name] = torch.ones_like(param)
    return masks


def aggregate_global_masks(
    client_masks_list,
    client_sizes,
    target_sparsity: float = 0.50,
    min_dim: int = 2,
):
    if not client_masks_list:
        raise ValueError("client_masks_list must not be empty")
    if len(client_masks_list) != len(client_sizes):
        raise ValueError(
            "client_masks_list and client_sizes must have the same length"
        )

    total = sum(client_sizes)
    if total <= 0:
        raise ValueError("client_sizes must have a positive total")

    global_masks = {}
    for name, reference_mask in client_masks_list[0].items():
        if not is_prunable_parameter(name, reference_mask, min_dim=min_dim):
            global_masks[name] = torch.ones_like(reference_mask)
            continue

        mask_scores = torch.zeros_like(
            reference_mask,
            dtype=torch.float32,
        )
        for client_masks, client_size in zip(
            client_masks_list,
            client_sizes,
        ):
            mask_scores.add_(
                client_masks[name].to(
                    device=mask_scores.device,
                    dtype=mask_scores.dtype,
                ),
                alpha=float(client_size / total),
            )

        flat_scores = mask_scores.reshape(-1)
        keep = max(
            1,
            int(round(flat_scores.numel() * (1.0 - target_sparsity))),
        )
        keep = min(keep, flat_scores.numel())
        flat_mask = torch.zeros_like(flat_scores)
        top_idx = torch.topk(
            flat_scores,
            k=keep,
            largest=True,
            sorted=False,
        ).indices
        flat_mask[top_idx] = 1.0
        global_masks[name] = (
            flat_mask.reshape_as(reference_mask).clone()
        )
    return global_masks


def apply_masks_to_model(model, masks) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))


def update_gradient_ema(
    model,
    grad_ema,
    gradients=None,
    decay: float = 0.9,
    min_dim: int = 2,
) -> None:
    for name, param in model.named_parameters():
        if not is_prunable_parameter(name, param, min_dim=min_dim):
            continue

        gradient = (
            param.grad
            if gradients is None
            else gradients.get(name)
        )
        if gradient is None:
            continue

        grad_abs = gradient.detach().abs()
        if name not in grad_ema:
            grad_ema[name] = grad_abs.clone()
        else:
            grad_ema[name].mul_(decay).add_(
                grad_abs,
                alpha=1.0 - decay,
            )


def prune_and_regrow(
    model,
    masks,
    grad_ema,
    prune_fraction: float = 0.20,
    scoring_grads=None,
    apply_to_model: bool = True,
    min_dim: int = 2,
):
    total_pruned = 0
    total_regrown = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if (
                not is_prunable_parameter(name, param, min_dim=min_dim)
                or name not in masks
            ):
                continue

            mask = masks[name].to(param.device)
            flat_mask = mask.reshape(-1).bool()
            active_idx = torch.nonzero(
                flat_mask,
                as_tuple=False,
            ).reshape(-1)
            inactive_idx = torch.nonzero(
                ~flat_mask,
                as_tuple=False,
            ).reshape(-1)

            num_prune = min(
                int(active_idx.numel() * prune_fraction),
                int(inactive_idx.numel()),
            )
            if num_prune <= 0:
                continue

            score_grad = (
                param.grad
                if scoring_grads is None
                else scoring_grads.get(name)
            )
            if score_grad is None:
                score_grad_abs = torch.zeros_like(param).reshape(-1)
            else:
                score_grad_abs = (
                    score_grad.detach().abs().reshape(-1).to(param.device)
                )

            importance = (
                param.detach().abs().reshape(-1) * score_grad_abs
            )
            prune_rel = torch.topk(
                importance[active_idx],
                k=num_prune,
                largest=False,
            ).indices
            prune_idx = active_idx[prune_rel]

            new_flat_mask = flat_mask.clone()
            new_flat_mask[prune_idx] = False

            grow_candidates = inactive_idx
            ema = grad_ema.get(name)
            if ema is None:
                ema = torch.zeros_like(param)
            ema = ema.to(param.device).reshape(-1)

            grow_rel = torch.topk(
                ema[grow_candidates],
                k=num_prune,
                largest=True,
            ).indices
            grow_idx = grow_candidates[grow_rel]
            new_flat_mask[grow_idx] = True

            masks[name] = (
                new_flat_mask.reshape_as(param).to(param.device).float()
            )
            if apply_to_model:
                param.mul_(masks[name])
                param.reshape(-1)[grow_idx] = 0.0

            total_pruned += num_prune
            total_regrown += int(grow_idx.numel())

    return total_pruned, total_regrown


def mask_statistics(
    previous_masks,
    current_masks,
    min_dim: int = 2,
):
    total = 0
    active = 0
    overlap = 0
    previous_active = 0

    for name, current in current_masks.items():
        if not is_prunable_parameter(name, current, min_dim=min_dim):
            continue

        current_bool = current.bool()
        total += int(current_bool.numel())
        active += int(current_bool.sum().item())

        if previous_masks is not None and name in previous_masks:
            previous_bool = (
                previous_masks[name].to(current.device).bool()
            )
            previous_active += int(previous_bool.sum().item())
            overlap += int(
                (previous_bool & current_bool).sum().item()
            )

    sparsity = 1.0 - active / max(total, 1)
    turnover = (
        0.0
        if previous_masks is None
        else 1.0 - overlap / max(previous_active, 1)
    )
    return float(sparsity), float(turnover), active, total


def mask_change_counts(
    previous_masks,
    current_masks,
    min_dim: int = 2,
):
    pruned = 0
    regrown = 0

    for name, current in current_masks.items():
        if not is_prunable_parameter(name, current, min_dim=min_dim):
            continue

        previous = previous_masks[name].to(current.device).bool()
        current_bool = current.bool()
        pruned += int((previous & ~current_bool).sum().item())
        regrown += int((~previous & current_bool).sum().item())

    return pruned, regrown
