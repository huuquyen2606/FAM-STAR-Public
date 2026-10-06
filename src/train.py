"""Local FAM-STAR optimization."""

from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .fam_star.topology import (
    apply_masks_to_model,
    prune_and_regrow,
    update_gradient_ema,
)


def get_params(model):
    return copy.deepcopy(model.state_dict())


def set_params(model, params) -> None:
    model.load_state_dict(params)


def compute_cb_weights(
    class_counts,
    num_classes: int,
    beta: float = 0.99,
    device="cpu",
):
    """Effective-number class weights normalized over locally present classes."""
    if not 0.0 <= beta < 1.0:
        raise ValueError(
            f"CB beta must satisfy 0 <= beta < 1; got {beta}"
        )

    weights = torch.zeros(
        num_classes,
        dtype=torch.float32,
        device=device,
    )
    for class_id, count in class_counts.items():
        if count > 0:
            weights[int(class_id)] = (
                (1.0 - beta)
                / (1.0 - (beta ** int(count)))
            )

    present_mask = weights > 0
    if present_mask.any():
        weights[present_mask] = (
            weights[present_mask]
            / weights[present_mask].sum()
            * present_mask.sum()
        )
    return weights


def local_train(
    model,
    loader,
    client_y,
    epochs: int,
    lr: float,
    device,
    global_params,
    global_masks,
    num_classes: int,
    proximal_mu: float,
    cb_beta: float,
    prune_fraction: float,
    missing_class_scale: float,
    prototype_lambda: float,
    prune_param_min_dim: int = 2,
    global_prototypes=None,
    make_topology_proposal: bool = True,
):
    """One FAM-STAR client's local optimization and next-topology proposal."""
    local_masks = {
        name: mask.detach().clone()
        for name, mask in global_masks.items()
    }
    apply_masks_to_model(model, local_masks)

    model.train()
    optimizer = optim.Adamax(
        model.parameters(),
        lr=lr,
    )

    class_counts = Counter(
        int(label) for label in client_y
    )
    cb_weights = compute_cb_weights(
        class_counts,
        num_classes=num_classes,
        beta=cb_beta,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(
        weight=cb_weights,
    )

    present_classes = np.array(
        sorted(class_counts),
        dtype=np.int64,
    )
    present_classes_t = torch.as_tensor(
        present_classes,
        dtype=torch.long,
        device=device,
    )

    if not 0.0 <= float(missing_class_scale) <= 1.0:
        raise ValueError(
            "missing_class_scale must be in [0, 1]"
        )

    rs_class_scale = torch.full(
        (num_classes,),
        float(missing_class_scale),
        dtype=torch.float32,
        device=device,
    )
    rs_class_scale[present_classes_t] = 1.0

    global_prototypes = global_prototypes or {}
    prototype_by_class = {
        int(class_id): prototype
        for class_id, prototype in global_prototypes.items()
    }
    prototype_class_ids = sorted(
        class_id
        for class_id in prototype_by_class
        if 0 <= class_id < num_classes
    )
    prototype_enabled = (
        bool(prototype_class_ids)
        and float(prototype_lambda) != 0.0
    )

    if prototype_enabled:
        prototype_targets = torch.stack(
            [
                prototype_by_class[class_id].to(
                    device=device,
                    dtype=torch.float32,
                )
                for class_id in prototype_class_ids
            ]
        )
        prototype_lookup = torch.full(
            (num_classes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        prototype_lookup[
            torch.as_tensor(
                prototype_class_ids,
                dtype=torch.long,
                device=device,
            )
        ] = torch.arange(
            len(prototype_class_ids),
            dtype=torch.long,
            device=device,
        )
        global_head_labels = torch.as_tensor(
            prototype_class_ids,
            dtype=torch.long,
            device=device,
        )
    else:
        prototype_targets = None
        prototype_lookup = None
        global_head_labels = None

    named_params = list(model.named_parameters())
    trainable_params = [
        param for _, param in named_params
    ]

    grad_ema = {}
    last_cb_gradients = {}

    running_loss = 0.0
    running_ce_loss = 0.0
    running_prox_loss = 0.0
    running_proto_loss = 0.0
    running_global_head_ce_loss = 0.0
    running_global_head_ce_active_classes = 0.0
    running_gradient_norm = 0.0

    pruned_count = 0
    regrown_count = 0
    num_batches = 0

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            if prototype_enabled:
                logits, embeddings = model(
                    xb,
                    return_embedding=True,
                )
            else:
                logits = model(xb)
                embeddings = None

            # CB-RS: keep global label space but down-scale absent-class logits.
            restricted_logits = (
                logits * rs_class_scale.unsqueeze(0)
            )
            ce_loss = criterion(
                restricted_logits,
                yb,
            )

            if prototype_enabled:
                prototype_indices = prototype_lookup[yb]
                valid = prototype_indices >= 0
                if bool(valid.any().item()):
                    normalized_embeddings = (
                        nn.functional.normalize(
                            embeddings[valid],
                            dim=1,
                        )
                    )
                    normalized_targets = (
                        nn.functional.normalize(
                            prototype_targets[
                                prototype_indices[valid]
                            ],
                            dim=1,
                        )
                    )
                    proto_loss = (
                        1.0
                        - (
                            normalized_embeddings
                            * normalized_targets
                        ).sum(dim=1)
                    ).mean()
                else:
                    proto_loss = torch.zeros(
                        (),
                        device=device,
                    )
            else:
                proto_loss = torch.zeros(
                    (),
                    device=device,
                )

            if (
                prototype_enabled
                and global_head_labels is not None
            ):
                global_head_features = (
                    prototype_targets.detach()
                )
                global_head_logits = model.fc_out(
                    global_head_features
                )
                global_head_ce_loss = (
                    nn.functional.cross_entropy(
                        global_head_logits,
                        global_head_labels,
                    )
                )
                global_head_active_classes = float(
                    global_head_labels.numel()
                )
            else:
                global_head_ce_loss = torch.zeros(
                    (),
                    device=device,
                )
                global_head_active_classes = 0.0

            proximal_term = torch.zeros(
                (),
                device=device,
            )
            for name, param in model.named_parameters():
                global_param = (
                    global_params[name]
                    .to(device)
                    .detach()
                )
                layer_mask = local_masks.get(name)
                if layer_mask is not None:
                    diff = (
                        layer_mask.to(device)
                        * (param - global_param)
                    )
                else:
                    diff = param - global_param
                proximal_term = (
                    proximal_term
                    + diff.pow(2).sum()
                )

            prox_loss = (
                proximal_mu / 2.0
            ) * proximal_term

            head_aux_loss = (
                float(prototype_lambda)
                * global_head_ce_loss
            )
            loss = (
                ce_loss
                + prox_loss
                + float(prototype_lambda) * proto_loss
                + head_aux_loss
            )

            # Only CB-RS CE supplies topology evidence.
            cb_grad_tuple = torch.autograd.grad(
                ce_loss,
                trainable_params,
                retain_graph=True,
                allow_unused=True,
            )
            cb_gradients = {
                name: (
                    gradient.detach()
                    if gradient is not None
                    else None
                )
                for (name, _), gradient in zip(
                    named_params,
                    cb_grad_tuple,
                )
            }
            last_cb_gradients = cb_gradients

            loss.backward()

            gradient_sq = torch.zeros(
                (),
                device=device,
            )
            for param in model.parameters():
                if param.grad is not None:
                    gradient_sq = (
                        gradient_sq
                        + param.grad.detach().pow(2).sum()
                    )
            running_gradient_norm += float(
                torch.sqrt(gradient_sq).item()
            )

            update_gradient_ema(
                model,
                grad_ema,
                gradients=cb_gradients,
                decay=0.9,
                min_dim=prune_param_min_dim,
            )

            optimizer.step()
            apply_masks_to_model(
                model,
                local_masks,
            )

            running_loss += float(
                loss.detach().item()
            )
            running_ce_loss += float(
                ce_loss.detach().item()
            )
            running_prox_loss += float(
                prox_loss.detach().item()
            )
            running_proto_loss += float(
                proto_loss.detach().item()
            )
            running_global_head_ce_loss += float(
                global_head_ce_loss.detach().item()
            )
            running_global_head_ce_active_classes += (
                global_head_active_classes
            )
            num_batches += 1

    # Topology is fixed during optimization; propose the next mask afterward.
    proposal_masks = None
    if make_topology_proposal:
        proposal_masks = {
            name: mask.detach().clone()
            for name, mask in local_masks.items()
        }
        if num_batches > 0:
            pruned_count, regrown_count = (
                prune_and_regrow(
                    model,
                    proposal_masks,
                    grad_ema,
                    prune_fraction=prune_fraction,
                    scoring_grads=last_cb_gradients,
                    apply_to_model=False,
                    min_dim=prune_param_min_dim,
                )
            )

    denom = max(num_batches, 1)
    return (
        get_params(model),
        local_masks,
        proposal_masks,
        {
            "train_loss": running_loss / denom,
            "ce_loss": running_ce_loss / denom,
            "prox_loss": running_prox_loss / denom,
            "proto_loss": running_proto_loss / denom,
            "global_head_ce_loss": (
                running_global_head_ce_loss / denom
            ),
            "global_head_ce_active_classes": (
                running_global_head_ce_active_classes
                / denom
            ),
            "global_head_ce_lambda": float(
                prototype_lambda
            ),
            "gradient_norm": (
                running_gradient_norm / denom
            ),
            "pruned_weights": pruned_count,
            "regrown_weights": regrown_count,
            "topology_proposed": int(
                proposal_masks is not None
            ),
            "restricted_softmax_alpha": float(
                missing_class_scale
            ),
            "present_class_count": int(
                len(present_classes)
            ),
            "missing_class_count": int(
                num_classes - len(present_classes)
            ),
        },
    )
