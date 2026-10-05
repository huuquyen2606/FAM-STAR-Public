"""Checkpoint I/O for FAM-STAR."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def save_round_checkpoint(
    path: str,
    round_id: int,
    model,
    global_masks,
    pending_global_masks,
    client_residuals,
    global_prototypes,
    round_metrics,
) -> None:
    """Save the same full per-round state carried by the notebook."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    weights = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    checkpoint_payload = {
        "round": int(round_id),
        "model_state": weights,
        "global_masks": {
            name: mask.detach().cpu()
            for name, mask in global_masks.items()
        },
        "pending_global_masks": (
            {
                name: mask.detach().cpu()
                for name, mask in pending_global_masks.items()
            }
            if pending_global_masks is not None
            else None
        ),
        "client_residuals": {
            int(cid): {
                name: residual.detach().cpu()
                for name, residual in residuals.items()
            }
            for cid, residuals in client_residuals.items()
        },
        "global_prototypes": {
            int(class_id): prototype.detach().cpu()
            for class_id, prototype in global_prototypes.items()
        },
        "metrics": dict(round_metrics),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    torch.save(checkpoint_payload, path)
