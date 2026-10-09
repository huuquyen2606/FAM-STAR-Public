"""End-to-end FAM-STAR federated training pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .data.dataset import (
    compute_client_class_statistics,
    load_prepared_dataset,
)
from .evaluate import evaluate_model
from .fam_star.aggregation import (
    apply_role_specific_aggregation,
    compute_head_aggregation_weights,
    compute_shared_aggregation_weights,
)
from .fam_star.compression import (
    build_payload_event,
    combine_delta_and_prototype_payload,
    compress_sparse_delta,
    summarize_sparse_model_payload,
)
from .fam_star.prototype import (
    aggregate_prototypes,
    compute_local_prototypes,
    quantize_prototype_package,
)
from .fam_star.topology import (
    aggregate_global_masks,
    apply_masks_to_model,
    init_sparse_masks,
    mask_change_counts,
    mask_statistics,
)
from .models.hybrid_blstm_gru import HybridBLSTM_GRU
from .train import get_params, local_train, set_params
from .utils.io import save_round_checkpoint
from .utils.metrics import export_results
from .utils.seed import set_seed


@dataclass
class FrameworkConfig:
    """FAM-STAR configuration derived from FAM_STAR_10clients.ipynb."""

    data_dir: str = ""
    output_dir: str = "fam_star_results"
    num_clients: int = 10
    num_rounds: int = 50
    num_epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 0.002

    # Local objective.
    fedprox_mu: float = 0.01
    cb_beta: float = 0.90
    missing_class_scale: float = 0.10
    prototype_lambda: float = 0.05

    # Dynamic sparse topology.
    total_sparsity: float = 0.50
    prune_fraction: float = 0.05
    prune_param_min_dim: int = 2

    # Role-specific server aggregation.
    aggregation_gamma: float = 0.50
    aggregation_eps: float = 1e-12
    head_weight_name: str = "fc_out.weight"
    head_bias_name: str = "fc_out.bias"

    # Compact exchange.
    int8_eps: float = 1e-8
    int8_scale_bytes: int = 4
    bitpack_order: str = "little"
    prototype_eps: float = 1e-8

    # Model.
    blstm_hidden: int = 300
    gru_hidden: int = 100
    dense_hidden: int = 80
    dropout: float = 0.3

    seed: int = 42
    deterministic: bool = True
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    )

    def __post_init__(self) -> None:
        if not self.data_dir:
            raise ValueError(
                "data_dir must point to the prepared CICAndMal2020 client folder"
            )
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if not 0.0 <= self.total_sparsity < 1.0:
            raise ValueError("total_sparsity must be in [0, 1)")
        if not 0.0 <= self.prune_fraction <= 1.0:
            raise ValueError("prune_fraction must be in [0, 1]")
        if not 0.0 <= self.missing_class_scale <= 1.0:
            raise ValueError(
                "missing_class_scale must be in [0, 1]"
            )


def build_base_model(
    input_size: int,
    num_classes: int,
    config: FrameworkConfig,
) -> HybridBLSTM_GRU:
    return HybridBLSTM_GRU(
        input_size=input_size,
        num_classes=num_classes,
        blstm_hidden=config.blstm_hidden,
        gru_hidden=config.gru_hidden,
        dense_hidden=config.dense_hidden,
        dropout=config.dropout,
    ).to(config.device)


def run_training_pipeline(
    config: FrameworkConfig,
) -> dict[str, Any]:
    """Train FAM-STAR while preserving the notebook's algorithmic sequence."""
    os.makedirs(config.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(
        config.output_dir,
        "checkpoints",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    set_seed(
        config.seed,
        deterministic=config.deterministic,
    )

    data = load_prepared_dataset(
        data_dir=config.data_dir,
        expected_num_clients=config.num_clients,
        batch_size=config.batch_size,
    )

    (
        client_class_counts,
        _client_sample_counts,
        sample_weights,
        global_class_counts,
    ) = compute_client_class_statistics(
        data.client_y,
        data.num_classes,
    )

    shared_aggregation_weights = (
        compute_shared_aggregation_weights(
            client_class_counts,
            sample_weights,
            gamma=config.aggregation_gamma,
            eps=config.aggregation_eps,
        )
    )
    head_aggregation_weights = (
        compute_head_aggregation_weights(
            client_class_counts,
            global_class_counts,
            sample_weights,
            gamma=config.aggregation_gamma,
            eps=config.aggregation_eps,
        )
    )

    global_model = build_base_model(
        input_size=data.num_features,
        num_classes=data.num_classes,
        config=config,
    )
    global_masks = init_sparse_masks(
        global_model,
        target_sparsity=config.total_sparsity,
        min_dim=config.prune_param_min_dim,
    )
    apply_masks_to_model(
        global_model,
        global_masks,
    )

    metrics_log = []
    payload_events = []
    client_residuals = {
        cid: {}
        for cid in range(config.num_clients)
    }
    global_prototypes = {}
    pending_global_masks = None

    print(
        f"Starting FAM-STAR: {config.num_rounds} rounds | "
        f"{config.num_clients} clients | "
        f"batch={config.batch_size} | "
        f"device={config.device}"
    )

    for round_id in range(
        1,
        config.num_rounds + 1,
    ):
        # Agreed topology becomes active only at the next round.
        if pending_global_masks is not None:
            global_masks = pending_global_masks
            apply_masks_to_model(
                global_model,
                global_masks,
            )
            pending_global_masks = None

        if config.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(
                config.device
            )

        global_params = get_params(global_model)
        global_prototypes_for_clients = {
            int(class_id): (
                prototype.detach()
                .cpu()
                .to(torch.float32)
                .clone()
            )
            for class_id, prototype
            in global_prototypes.items()
        }

        download_payload = summarize_sparse_model_payload(
            global_params,
            global_masks,
            global_prototypes=(
                global_prototypes_for_clients
            ),
            min_dim=config.prune_param_min_dim,
            bitorder=config.bitpack_order,
        )

        local_start = time.perf_counter()

        client_delta_list = []
        client_sizes = []
        client_train_stats = []
        client_proposed_masks = []
        client_prototype_packages = []

        for cid in range(config.num_clients):
            local_model = build_base_model(
                input_size=data.num_features,
                num_classes=data.num_classes,
                config=config,
            )
            set_params(
                local_model,
                global_params,
            )

            downlink_payload_type = (
                "sparse_fp32_model_bitpacked_mask_"
                "plus_global_prototypes_fp32"
                if global_prototypes_for_clients
                else "sparse_fp32_model_bitpacked_mask"
            )
            payload_events.append(
                build_payload_event(
                    round_id,
                    cid,
                    "downlink",
                    downlink_payload_type,
                    download_payload,
                )
            )

            (
                local_params,
                local_masks,
                proposed_masks,
                client_stats,
            ) = local_train(
                model=local_model,
                loader=data.client_trainloaders[cid],
                client_y=data.client_y[cid],
                epochs=config.num_epochs,
                lr=config.learning_rate,
                device=config.device,
                global_params=global_params,
                global_masks=global_masks,
                num_classes=data.num_classes,
                proximal_mu=config.fedprox_mu,
                cb_beta=config.cb_beta,
                prune_fraction=config.prune_fraction,
                missing_class_scale=(
                    config.missing_class_scale
                ),
                prototype_lambda=(
                    config.prototype_lambda
                ),
                prune_param_min_dim=(
                    config.prune_param_min_dim
                ),
                global_prototypes=(
                    global_prototypes_for_clients
                ),
                make_topology_proposal=True,
            )

            (
                local_prototypes,
                local_supports,
                local_dispersions,
            ) = compute_local_prototypes(
                local_model,
                data.client_proto_loaders[cid],
                config.device,
                data.num_classes,
            )

            prototype_result = (
                quantize_prototype_package(
                    local_prototypes,
                    local_supports,
                    local_dispersions,
                    eps=config.prototype_eps,
                    scale_bytes=(
                        config.int8_scale_bytes
                    ),
                )
            )
            client_prototype_packages.append(
                {
                    "prototypes": (
                        prototype_result[
                            "prototypes"
                        ]
                    ),
                    "supports": (
                        prototype_result[
                            "supports"
                        ]
                    ),
                    "dispersions": (
                        prototype_result[
                            "dispersions"
                        ]
                    ),
                }
            )

            compressed = compress_sparse_delta(
                local_state=local_params,
                global_state=global_params,
                masks=local_masks,
                proposal_masks=proposed_masks,
                previous_residuals=(
                    client_residuals[cid]
                ),
                eps=config.int8_eps,
                min_dim=(
                    config.prune_param_min_dim
                ),
                bitorder=config.bitpack_order,
                scale_bytes=(
                    config.int8_scale_bytes
                ),
            )

            client_residuals[cid] = (
                compressed["residuals"]
            )
            client_delta_list.append(
                compressed[
                    "dequantized_delta"
                ]
            )
            client_sizes.append(
                len(data.client_y[cid])
            )
            client_train_stats.append(
                client_stats
            )

            if proposed_masks is not None:
                client_proposed_masks.append(
                    {
                        name: mask.detach().clone()
                        for name, mask
                        in proposed_masks.items()
                    }
                )

            _, uplink_summary = (
                combine_delta_and_prototype_payload(
                    compressed,
                    prototype_result,
                )
            )
            uplink_payload_type = (
                "sparse_int8_delta_bitpacked_support_"
                "error_feedback_plus_bitpacked_next_mask_"
                "plus_reliable_prototypes"
                if proposed_masks is not None
                else
                "sparse_int8_delta_bitpacked_support_"
                "error_feedback_plus_reliable_prototypes"
            )
            payload_events.append(
                build_payload_event(
                    round_id,
                    cid,
                    "uplink",
                    uplink_payload_type,
                    uplink_summary,
                    error_norm=compressed[
                        "error_norm"
                    ],
                )
            )

        local_time_sec = (
            time.perf_counter()
            - local_start
        )

        server_start = time.perf_counter()

        updated_params, aggregation_stats = (
            apply_role_specific_aggregation(
                global_params=global_params,
                client_delta_list=client_delta_list,
                shared_client_weights=(
                    shared_aggregation_weights
                ),
                head_client_weights=(
                    head_aggregation_weights
                ),
                head_weight_name=(
                    config.head_weight_name
                ),
                head_bias_name=(
                    config.head_bias_name
                ),
                eps=config.aggregation_eps,
            )
        )
        set_params(
            global_model,
            updated_params,
        )

        (
            global_prototypes,
            prototype_stats,
        ) = aggregate_prototypes(
            client_prototype_packages,
            previous_prototypes=(
                global_prototypes
            ),
            eps=config.prototype_eps,
        )

        # Current round is evaluated with the current topology.
        apply_masks_to_model(
            global_model,
            global_masks,
        )

        _, _, _, prunable_params = (
            mask_statistics(
                None,
                global_masks,
                min_dim=(
                    config.prune_param_min_dim
                ),
            )
        )

        if (
            len(client_proposed_masks)
            != len(client_sizes)
        ):
            raise RuntimeError(
                "Every participating client must send one "
                "FAM-STAR topology proposal"
            )

        next_global_masks = (
            aggregate_global_masks(
                client_proposed_masks,
                client_sizes,
                target_sparsity=(
                    config.total_sparsity
                ),
                min_dim=(
                    config.prune_param_min_dim
                ),
            )
        )

        (
            next_mask_sparsity,
            _,
            _,
            _,
        ) = mask_statistics(
            global_masks,
            next_global_masks,
            min_dim=config.prune_param_min_dim,
        )
        (
            next_mask_pruned_weights,
            next_mask_regrown_weights,
        ) = mask_change_counts(
            global_masks,
            next_global_masks,
            min_dim=config.prune_param_min_dim,
        )

        if (
            next_mask_pruned_weights
            != next_mask_regrown_weights
        ):
            raise AssertionError(
                "FAM-STAR mask vote must preserve active count"
            )

        sparsity_tolerance = (
            1.0 / max(prunable_params, 1)
        )
        if (
            abs(
                next_mask_sparsity
                - config.total_sparsity
            )
            > sparsity_tolerance
        ):
            raise AssertionError(
                "FAM-STAR next-round mask violates target sparsity"
            )

        server_time_sec = (
            time.perf_counter()
            - server_start
        )

        metrics = evaluate_model(
            global_model,
            data.global_testloader,
            config.device,
            data.num_classes,
        )

        total_samples = max(
            sum(client_sizes),
            1,
        )
        train_loss = float(
            sum(
                stats["train_loss"] * size
                for stats, size in zip(
                    client_train_stats,
                    client_sizes,
                )
            )
            / total_samples
        )

        if config.device.type == "cuda":
            peak_vram_mb = (
                torch.cuda.max_memory_allocated(
                    config.device
                )
                / (1024 ** 2)
            )
        else:
            peak_vram_mb = 0.0

        round_metrics = {
            "round": round_id,
            "accuracy": metrics["accuracy"],
            "precision_macro": (
                metrics["precision_macro"]
            ),
            "precision_micro": (
                metrics["precision_micro"]
            ),
            "precision_weighted": (
                metrics["precision_weighted"]
            ),
            "recall_macro": (
                metrics["recall_macro"]
            ),
            "recall_micro": (
                metrics["recall_micro"]
            ),
            "recall_weighted": (
                metrics["recall_weighted"]
            ),
            "f1_macro": metrics["f1_macro"],
            "f1_micro": metrics["f1_micro"],
            "f1_weighted": (
                metrics["f1_weighted"]
            ),
            "worst_class_f1": (
                metrics["worst_class_f1"]
            ),
            "balanced_accuracy": (
                metrics["balanced_accuracy"]
            ),
            "train_loss": train_loss,
            "local_time_sec": round(
                local_time_sec,
                2,
            ),
            "server_time_sec": round(
                server_time_sec,
                2,
            ),
            "peak_vram_mb": round(
                float(peak_vram_mb),
                2,
            ),
        }
        metrics_log.append(round_metrics)

        # Next topology is pending until the start of the next round.
        pending_global_masks = (
            next_global_masks
            if round_id < config.num_rounds
            else None
        )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"round_{round_id:02d}_famstar.pt",
        )
        save_round_checkpoint(
            path=checkpoint_path,
            round_id=round_id,
            model=global_model,
            global_masks=global_masks,
            pending_global_masks=(
                pending_global_masks
            ),
            client_residuals=(
                client_residuals
            ),
            global_prototypes=(
                global_prototypes
            ),
            round_metrics=round_metrics,
        )

        print(
            f"Round {round_id:02d}/{config.num_rounds} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1_W={metrics['f1_weighted']:.4f} | "
            f"F1_M={metrics['f1_macro']:.4f} | "
            f"Local={local_time_sec:.1f}s | "
            f"Server={server_time_sec:.1f}s"
        )

    final_metrics = evaluate_model(
        global_model,
        data.global_testloader,
        config.device,
        data.num_classes,
    )
    exported = export_results(
        metrics_log=metrics_log,
        payload_events=payload_events,
        final_metrics=final_metrics,
        classes=data.classes,
        output_dir=config.output_dir,
        num_rounds=config.num_rounds,
    )

    return {
        "model": global_model,
        "global_masks": global_masks,
        "global_prototypes": global_prototypes,
        "metrics_log": metrics_log,
        "payload_events": payload_events,
        "final_metrics": final_metrics,
        "aggregation_stats": aggregation_stats,
        "prototype_stats": prototype_stats,
        "exports": exported,
    }
