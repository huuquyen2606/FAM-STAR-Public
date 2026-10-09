"""Wireless quantized FL port of Quantization/Wireless/{10,20,50} clients.

Preserves the paper's continuous KKT relaxation, integer re-optimization,
Lambert-W branches and stochastic scalar quantization. Uplink byte accounting
is analytical, not a packed codec; downlink is measured torch serialization.
"""

from __future__ import annotations

import os

# Must precede torch imports for deterministic CUDA RNN operations.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import lambertw

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
METHOD_NAME = "Wireless"
METHOD_SLUG = "wireless"
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

WIRELESS_EPSILON = 0.01
QUANTIZATION_METADATA_BITS = 64
MIN_QUANTIZATION_BITS = 1
MAX_QUANTIZATION_BITS = 16
WIRELESS_SOLVER_MAXITER = 300
WIRELESS_BANDWIDTH_HZ = 0.3e6
NOISE_SPECTRAL_DENSITY_W_HZ = 10 ** ((-174 - 30) / 10)
CLIENT_CPU_FREQUENCY_HZ = 1.5e9
CPU_ENERGY_COEFFICIENT = 1e-27
CLIENT_ENERGY_BUDGET_J = 0.3
PATH_LOSS_EXPONENT = 3.75
WIRELESS_RANDOM_SEED = 42
CLIENT_COMPUTE_CYCLES_RANGE = (10.0, 40.0)
CLIENT_DISTANCE_RANGE_M = (0.0, 1000.0)
# The 50-client source deletes each preceding checkpoint; 10/20 retain all.
LATEST_CHECKPOINT_ONLY_CLIENT_COUNTS = {50}


def quantize_tensor(tensor, B, abs_min=None, abs_max=None, generator=None):
    # Stochastic scalar quantization with one range shared by the whole
    # client update. The sign is one extra bit per coordinate.
    if not torch.is_floating_point(tensor):
        return tensor.float()
    if B >= 32:
        return tensor
    abs_tensor = torch.abs(tensor)
    sign = torch.sign(tensor)
    t_min = (
        torch.min(abs_tensor)
        if abs_min is None
        else torch.as_tensor(abs_min, device=tensor.device, dtype=tensor.dtype)
    )
    t_max = (
        torch.max(abs_tensor)
        if abs_max is None
        else torch.as_tensor(abs_max, device=tensor.device, dtype=tensor.dtype)
    )
    num_intervals = 2**B - 1
    interval_width = (t_max - t_min) / num_intervals
    if (not torch.isfinite(interval_width).item()) or interval_width.item() <= 1e-12:
        return tensor
    normalized = torch.clamp(
        (abs_tensor - t_min) / interval_width, 0.0, float(num_intervals)
    )
    lower_idx = torch.floor(normalized).clamp(0, num_intervals - 1)
    s_lower = t_min + lower_idx * interval_width
    s_upper = s_lower + interval_width
    prob_upper = torch.clamp(normalized - lower_idx, 0.0, 1.0)
    if generator is None:
        random_values = torch.rand_like(tensor)
    else:
        random_values = torch.rand(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )
    use_upper = random_values < prob_upper
    quantized_abs = torch.where(use_upper, s_upper, s_lower)
    return quantized_abs * sign


def summarize_wireless_payload(
    raw_delta, quantized_delta, bits, metadata_bits, model_update_dim, float_keys
):
    float_keys = list(float_keys)
    raw_bytes = int(
        sum(
            raw_delta[key].numel() * raw_delta[key].element_size() for key in float_keys
        )
    )
    nnz = int(
        sum(torch.count_nonzero(quantized_delta[key]).item() for key in float_keys)
    )
    values_bytes = int(math.ceil(model_update_dim * (bits + 1) / 8.0))
    scales_bytes = int(math.ceil(metadata_bits / 8.0))
    serialized_bytes = int(values_bytes + scales_bytes)
    return {
        "raw_bytes": raw_bytes,
        "serialized_bytes": serialized_bytes,
        "values_bytes": values_bytes,
        "indices_bytes": 0,
        "scales_bytes": scales_bytes,
        "mask_bytes": 0,
        "prototype_bytes": 0,
        "metadata_bytes": 0,
        "nnz": nnz,
        "total_params": int(model_update_dim),
        "sparsity": float(1.0 - nnz / max(model_update_dim, 1)),
    }


def payload_error_norm(reference, reconstructed, keys):
    squared = 0.0
    for key in keys:
        diff = reference[key].float() - reconstructed[key].float()
        squared += float(torch.sum(diff * diff).item())
    return float(math.sqrt(squared))


def solve_uplink_time_lambert(message_bits, gain, energy):
    # Invert C1 exactly with the W_{-1} branch of Lambert-W.
    message_bits, gain, energy = np.broadcast_arrays(
        np.asarray(message_bits, dtype=np.float64),
        np.asarray(gain, dtype=np.float64),
        np.asarray(energy, dtype=np.float64),
    )
    bits_flat, gain_flat, energy_flat = (
        message_bits.ravel(),
        gain.ravel(),
        energy.ravel(),
    )
    result_flat = np.full(bits_flat.shape, np.inf, dtype=np.float64)
    valid = (bits_flat > 0.0) & (gain_flat > 0.0) & (energy_flat > 0.0)
    if np.any(valid):
        x = (
            bits_flat[valid]
            * np.log(2.0)
            * NOISE_SPECTRAL_DENSITY_W_HZ
            / (gain_flat[valid] * energy_flat[valid])
        )
        valid_x = np.isfinite(x) & (x < 1.0)
        if np.any(valid_x):
            w_minus_one = np.real(lambertw(-x[valid_x] * np.exp(-x[valid_x]), k=-1))
            denominator = -(w_minus_one + x[valid_x])
            valid_denominator = np.isfinite(denominator) & (denominator > 0.0)
            valid_indices = np.flatnonzero(valid)[valid_x]
            result_flat[valid_indices[valid_denominator]] = (
                bits_flat[valid_indices[valid_denominator]]
                * np.log(2.0)
                / (WIRELESS_BANDWIDTH_HZ * denominator[valid_denominator])
            )
    result = result_flat.reshape(message_bits.shape)
    return float(result) if result.ndim == 0 else result


def _paper_uplink_time_from_multiplier(lambda_1, gain, energy):
    # Paper Eq. (32): W_0 is the principal Lambert-W branch.
    lambda_1, gain, energy = np.broadcast_arrays(
        np.asarray(lambda_1, dtype=np.float64),
        np.asarray(gain, dtype=np.float64),
        np.asarray(energy, dtype=np.float64),
    )
    result = np.full(lambda_1.shape, np.inf, dtype=np.float64)
    valid = (lambda_1 > 0.0) & (gain > 0.0) & (energy > 0.0)
    if np.any(valid):
        c = np.log(2.0) / (WIRELESS_BANDWIDTH_HZ * lambda_1[valid])
        psi = -np.exp(-1.0 - c)
        w_zero = np.real(lambertw(psi, k=0))
        denominator = 1.0 + w_zero
        good = np.isfinite(w_zero) & (denominator > 0.0)
        values = np.full_like(w_zero, np.inf, dtype=np.float64)
        values[good] = (
            -gain[valid][good]
            * energy[valid][good]
            / (WIRELESS_BANDWIDTH_HZ * NOISE_SPECTRAL_DENSITY_W_HZ)
            * w_zero[good]
            / denominator[good]
        )
        result[valid] = values
    return float(result) if result.ndim == 0 else result


def _paper_lambda_1_from_uplink_time(uplink_time, gain, energy):
    # Eq. (31), rearranged to obtain lambda_1 from l_up.
    ratio = (
        gain
        * energy
        / (uplink_time * WIRELESS_BANDWIDTH_HZ * NOISE_SPECTRAL_DENSITY_W_HZ)
    )
    q = np.log1p(ratio) - ratio / (1.0 + ratio)
    return np.log(2.0) / (WIRELESS_BANDWIDTH_HZ * q)


def allocate_bits_joint(
    delta_scales,
    client_sizes,
    epsilon,
    local_epochs,
    min_bits,
    max_bits,
    solver_maxiter,
    *,
    model_update_dim,
    client_compute_cycles_per_sample,
    channel_power_gains,
):
    # Paper IV-B: continuous KKT relaxation, dual lambda_3 search,
    # ceil B_n, and the fixed-integer re-optimization in (36).
    n = len(client_sizes)
    if n == 0 or epsilon <= 0.0 or min_bits < 1 or max_bits < min_bits:
        raise ValueError("Invalid Wireless KKT allocation parameters.")
    weights = np.array(client_sizes, dtype=np.float64, copy=True)
    if np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("client_sizes must be positive and finite.")
    weights /= weights.sum()
    delta = np.asarray(delta_scales, dtype=np.float64)
    if (
        delta.shape != weights.shape
        or np.any(delta < 0.0)
        or not np.isfinite(delta).all()
    ):
        raise ValueError("delta_scales must be non-negative and match client_sizes.")
    samples = np.asarray(client_sizes, dtype=np.float64)
    cycles = local_epochs * client_compute_cycles_per_sample * samples
    energy_coeff = CPU_ENERGY_COEFFICIENT * cycles**3
    lc_min = float(np.max(cycles / CLIENT_CPU_FREQUENCY_HZ))
    lc_energy = float(np.max(np.sqrt(energy_coeff / CLIENT_ENERGY_BUDGET_J)))
    lc_lower = max(lc_min, lc_energy) * (1.0 + 1e-9)
    bit_iterations = max(18, min(28, solver_maxiter // 12))
    dual_iterations = max(20, min(36, solver_maxiter // 10))
    compute_iterations = max(12, min(28, solver_maxiter // 12))

    def weighted_error(bits):
        levels = np.exp2(np.asarray(bits, dtype=np.float64)) - 1.0
        return float(np.sum(weights * delta**2 / levels**2))

    minimum_error = weighted_error(np.full(n, float(max_bits)))
    if minimum_error > epsilon + max(1e-12, epsilon * 1e-8):
        raise RuntimeError(
            "Paper KKT allocation infeasible: C3 cannot be met with max_bits."
        )

    def primal_at_lambda_3(lc, lambda_3):
        available_energy = CLIENT_ENERGY_BUDGET_J - energy_coeff / float(lc) ** 2
        if np.any(available_energy <= 0.0):
            return None
        lower = np.full(n, float(min_bits))
        upper = np.full(n, float(max_bits))

        def stationarity_residual(bits):
            message_bits = model_update_dim * (bits + 1.0) + QUANTIZATION_METADATA_BITS
            uplink_time = solve_uplink_time_lambert(
                message_bits, channel_power_gains, available_energy
            )
            lambda_1 = _paper_lambda_1_from_uplink_time(
                uplink_time, channel_power_gains, available_energy
            )
            levels = np.exp2(bits)
            residual = (
                model_update_dim * lambda_1
                - 2.0
                * lambda_3
                * np.log(2.0)
                * weights
                * delta**2
                * levels
                / (levels - 1.0) ** 3
            )
            return residual, uplink_time, lambda_1

        lower_residual, _, _ = stationarity_residual(lower)
        upper_residual, _, _ = stationarity_residual(upper)
        bits = lower.copy()
        upper_active = upper_residual <= 0.0
        bits[upper_active] = upper[upper_active]
        interior = (lower_residual < 0.0) & (upper_residual > 0.0)
        left, right = lower.copy(), upper.copy()
        for _ in range(bit_iterations):
            middle = (left + right) / 2.0
            middle_residual, _, _ = stationarity_residual(middle)
            move_right = interior & (middle_residual < 0.0)
            move_left = interior & ~move_right
            left[move_right] = middle[move_right]
            right[move_left] = middle[move_left]
        bits[interior] = (left[interior] + right[interior]) / 2.0
        _, uplink_time, lambda_1 = stationarity_residual(bits)
        error = weighted_error(bits)
        if not np.isfinite(error) or not np.all(np.isfinite(uplink_time)):
            return None
        return bits, uplink_time, error, available_energy, lambda_1, lambda_3

    def continuous_state(lc):
        state = primal_at_lambda_3(lc, 0.0)
        if state is None:
            return None
        if state[2] > epsilon:
            lambda_3_low, lambda_3_high = 0.0, 1.0
            high_state = None
            for _ in range(80):
                high_state = primal_at_lambda_3(lc, lambda_3_high)
                if high_state is not None and high_state[2] <= epsilon:
                    break
                lambda_3_high *= 10.0
            if high_state is None or high_state[2] > epsilon:
                return None
            for _ in range(dual_iterations):
                lambda_3_mid = (lambda_3_low + lambda_3_high) / 2.0
                middle_state = primal_at_lambda_3(lc, lambda_3_mid)
                if middle_state[2] <= epsilon:
                    lambda_3_high = lambda_3_mid
                else:
                    lambda_3_low = lambda_3_mid
            state = primal_at_lambda_3(lc, lambda_3_high)
        lambda_1 = state[4]
        uplink_time = state[1]
        available_energy = state[3]
        lambda_2_denominator = np.log(2.0) * (
            available_energy / (uplink_time * WIRELESS_BANDWIDTH_HZ)
            + NOISE_SPECTRAL_DENSITY_W_HZ / channel_power_gains
        )
        if np.any(lambda_2_denominator <= 0.0):
            return None
        lambda_2 = lambda_1 / lambda_2_denominator
        lc_from_eq27 = float((2.0 * np.sum(lambda_2 * energy_coeff)) ** (1.0 / 3.0))
        state = dict(
            bits=state[0],
            uplink_time=uplink_time,
            error=state[2],
            available_energy=available_energy,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=state[5],
            lc_from_eq27=lc_from_eq27,
            lc=float(lc),
        )
        return state

    lc_left = lc_lower * (1.0 + 1e-6)
    left_state = None
    for _ in range(12):
        left_state = continuous_state(lc_left)
        if left_state is not None:
            break
        lc_left = max(lc_left * 2.0, lc_lower * (1.0 + 1e-6))
    if left_state is None:
        raise RuntimeError("Paper KKT allocation has no feasible l_c.")

    if left_state["lc_from_eq27"] > lc_left:
        lc_right = max(lc_left * 100.0, lc_left + 1e-3)
        right_state = None
        for _ in range(12):
            right_state = continuous_state(lc_right)
            if right_state is not None and right_state["lc_from_eq27"] <= lc_right:
                break
            lc_right *= 2.0
        if right_state is None or right_state["lc_from_eq27"] > lc_right:
            raise RuntimeError("Paper KKT Eq. (27) did not bracket l_c.")
        for _ in range(compute_iterations):
            lc_middle = (lc_left + lc_right) / 2.0
            middle_state = continuous_state(lc_middle)
            if middle_state is None:
                lc_left = lc_middle
            elif middle_state["lc_from_eq27"] > lc_middle:
                lc_left, left_state = lc_middle, middle_state
            else:
                lc_right, right_state = lc_middle, middle_state
        continuous = continuous_state((lc_left + lc_right) / 2.0)
    else:
        continuous = left_state
    if continuous is None:
        raise RuntimeError("Paper KKT allocation failed for continuous relaxation.")

    bits = np.clip(
        np.ceil(continuous["bits"] - 1e-10).astype(int),
        min_bits,
        max_bits,
    )
    error_bound = weighted_error(bits)
    if error_bound > epsilon + max(1e-10, epsilon * 1e-8):
        raise RuntimeError("Paper KKT allocation violated C3 after ceil.")
    message_bits = model_update_dim * (bits + 1.0) + QUANTIZATION_METADATA_BITS

    def fixed_state(lc):
        available_energy = CLIENT_ENERGY_BUDGET_J - energy_coeff / float(lc) ** 2
        if np.any(available_energy <= 0.0):
            return None
        uplink_time = solve_uplink_time_lambert(
            message_bits, channel_power_gains, available_energy
        )
        if not np.all(np.isfinite(uplink_time)):
            return None
        lambda_1 = _paper_lambda_1_from_uplink_time(
            uplink_time, channel_power_gains, available_energy
        )
        lambda_2_denominator = np.log(2.0) * (
            available_energy / (uplink_time * WIRELESS_BANDWIDTH_HZ)
            + NOISE_SPECTRAL_DENSITY_W_HZ / channel_power_gains
        )
        if np.any(lambda_2_denominator <= 0.0):
            return None
        lambda_2 = lambda_1 / lambda_2_denominator
        lc_from_eq27 = float((2.0 * np.sum(lambda_2 * energy_coeff)) ** (1.0 / 3.0))
        return dict(
            lc=float(lc),
            uplink_time=uplink_time,
            available_energy=available_energy,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lc_from_eq27=lc_from_eq27,
        )

    fixed_left = lc_lower * (1.0 + 1e-6)
    fixed_left_state = None
    for _ in range(12):
        fixed_left_state = fixed_state(fixed_left)
        if fixed_left_state is not None:
            break
        fixed_left = max(fixed_left * 2.0, lc_lower * (1.0 + 1e-6))
    if fixed_left_state is None:
        raise RuntimeError("Paper fixed-bit re-optimization has no feasible l_c.")

    if fixed_left_state["lc_from_eq27"] > fixed_left:
        fixed_right = max(fixed_left * 100.0, fixed_left + 1e-3)
        fixed_right_state = None
        for _ in range(12):
            fixed_right_state = fixed_state(fixed_right)
            if (
                fixed_right_state is not None
                and fixed_right_state["lc_from_eq27"] <= fixed_right
            ):
                break
            fixed_right *= 2.0
        if fixed_right_state is None or fixed_right_state["lc_from_eq27"] > fixed_right:
            raise RuntimeError("Paper fixed-bit Eq. (27) did not bracket l_c.")
        for _ in range(compute_iterations):
            lc_middle = (fixed_left + fixed_right) / 2.0
            middle_state = fixed_state(lc_middle)
            if middle_state is None:
                fixed_left = lc_middle
            elif middle_state["lc_from_eq27"] > lc_middle:
                fixed_left, fixed_left_state = lc_middle, middle_state
            else:
                fixed_right, fixed_right_state = lc_middle, middle_state
        fixed = fixed_state((fixed_left + fixed_right) / 2.0)
    else:
        fixed = fixed_left_state
    if fixed is None:
        raise RuntimeError("Paper fixed-bit re-optimization failed.")

    # Eq. (32) must reproduce the l_up obtained from Eq. (34).
    eq32_time = _paper_uplink_time_from_multiplier(
        fixed["lambda_1"], channel_power_gains, fixed["available_energy"]
    )
    relative_eq32_error = np.max(
        np.abs(eq32_time - fixed["uplink_time"])
        / np.maximum(fixed["uplink_time"], 1e-12)
    )
    if not np.isfinite(relative_eq32_error) or relative_eq32_error > 1e-5:
        raise RuntimeError("Paper Eq. (32) consistency check failed.")
    planned_compute_time = float(fixed["lc"])
    planned_round_time = planned_compute_time + float(np.sum(fixed["uplink_time"]))
    return bits.tolist(), float(error_bound), planned_compute_time, planned_round_time


def compute_wireless_round_cost(
    message_bits,
    client_sizes,
    local_epochs,
    common_compute_time=None,
    *,
    client_compute_cycles_per_sample,
    channel_power_gains,
):
    # TDMA model from the paper: a common computation phase followed by
    # orthogonal uplinks. The CPU runs at the smallest frequency that
    # makes all clients finish computation at the same l_c.
    sample_counts = np.asarray(client_sizes, dtype=np.float64)
    required_cycles = local_epochs * client_compute_cycles_per_sample * sample_counts
    raw_compute_time = required_cycles / CLIENT_CPU_FREQUENCY_HZ
    minimum_compute_time = float(np.max(raw_compute_time))
    if common_compute_time is None:
        common_compute_time = minimum_compute_time
    common_compute_time = max(float(common_compute_time), minimum_compute_time)
    effective_cpu_frequency = required_cycles / common_compute_time
    compute_energy = (
        CPU_ENERGY_COEFFICIENT * required_cycles * effective_cpu_frequency**2
    )
    available_uplink_energy = CLIENT_ENERGY_BUDGET_J - compute_energy

    # Solve C1 with equality for each client using the Lambert-W inverse:
    # l_up W log2(1 + g E_up/(l_up W N0)) = S_n.
    message_bits = np.asarray(message_bits, dtype=np.float64)
    uplink_time = np.asarray(
        [
            solve_uplink_time_lambert(bits, gain, energy)
            for bits, gain, energy in zip(
                message_bits, channel_power_gains, available_uplink_energy
            )
        ],
        dtype=np.float64,
    )

    finite_uplink = np.isfinite(uplink_time)
    uplink_rate = np.zeros_like(uplink_time)
    uplink_rate[finite_uplink] = (
        message_bits[finite_uplink] / uplink_time[finite_uplink]
    )
    uplink_energy = np.where(finite_uplink, available_uplink_energy, np.inf)
    total_energy = compute_energy + uplink_energy
    return {
        "minimum_compute_time_sec": minimum_compute_time,
        "compute_time_sec": common_compute_time,
        "compute_time_per_client_sec": np.full_like(
            required_cycles, common_compute_time
        ),
        "effective_cpu_frequency_hz": effective_cpu_frequency,
        "uplink_rate_bps": uplink_rate,
        "uplink_time_per_client_sec": uplink_time,
        "communication_time_sec": float(np.sum(uplink_time)),
        "compute_energy_per_client_joule": compute_energy,
        "uplink_energy_per_client_joule": uplink_energy,
        "total_energy_per_client_joule": total_energy,
        "total_energy_joule": float(np.sum(total_energy)),
        "round_time_sec": float(common_compute_time + np.sum(uplink_time)),
        "energy_constraint_satisfied": bool(
            np.all(finite_uplink & (total_energy <= CLIENT_ENERGY_BUDGET_J + 1e-9))
        ),
    }


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
    """Run source stochastic quantization and joint wireless allocation."""
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the source realization and its draw order: cycles, distance, fading.
    wireless_rng = np.random.default_rng(WIRELESS_RANDOM_SEED)
    client_compute_cycles_per_sample = wireless_rng.uniform(
        *CLIENT_COMPUTE_CYCLES_RANGE,
        size=num_clients,
    )
    client_distances_m = np.maximum(
        wireless_rng.uniform(*CLIENT_DISTANCE_RANGE_M, size=num_clients),
        1.0,
    )
    channel_power_gains = wireless_rng.exponential(
        scale=1.0, size=num_clients
    ) * client_distances_m ** (-PATH_LOSS_EXPONENT)
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
    wireless_metrics_log = []
    wireless_cost_log = []
    wireless_round_costs = []
    last_round_state = {}
    float_update_keys = {
        key
        for key, value in global_model.state_dict().items()
        if torch.is_floating_point(value)
    }
    model_update_dim = int(
        sum(global_model.state_dict()[key].numel() for key in float_update_keys)
    )

    print(
        f"Starting Federated Learning: {num_rounds} rounds | {num_clients} clients | batch={batch_size}\n"
    )

    for rnd in range(1, num_rounds + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        global_params = get_params(global_model)
        download_payload = summarize_payload(global_params)

        # Fixed quantization-error tolerance for the main comparison.
        epsilon = WIRELESS_EPSILON

        # 2. Local train and collect the complete floating-point model update
        local_t0 = time.perf_counter()
        # before assigning bits, because B_n is selected jointly across clients.
        client_raw_deltas = []
        client_delta_ranges = []
        client_delta_scales = []
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
            set_params(local_model, global_params)

            payload_events.append(
                build_payload_event(rnd, cid, "downlink", "model", download_payload)
            )

            params, client_train_loss = local_train(
                model=local_model,
                loader=client_trainloaders[cid],
                epochs=num_epochs,
                lr=learning_rate,
                device=device,
            )

            # Paper's Delta w_n is the full floating-point update vector.
            # Integer bookkeeping buffers (e.g. BN counters) are not model
            # coordinates and therefore do not consume quantization bits.
            raw_delta = {}
            all_diff_vals = []
            for key in float_update_keys:
                diff = params[key].float() - global_params[key].float()
                raw_delta[key] = diff
                all_diff_vals.append(torch.abs(diff).flatten())

            for key in global_params:
                if key not in float_update_keys:
                    raw_delta[key] = torch.zeros_like(
                        global_params[key], dtype=torch.float32
                    )

            all_diff_tensor = torch.cat(all_diff_vals)
            diff_min = torch.min(all_diff_tensor).item()
            diff_max = torch.max(all_diff_tensor).item()
            d_total = all_diff_tensor.numel()
            delta_n = (math.sqrt(d_total) / 2.0) * (diff_max - diff_min)

            client_raw_deltas.append(raw_delta)
            client_delta_ranges.append((diff_min, diff_max))
            client_delta_scales.append(delta_n)
            client_sizes.append(len(client_y_list[cid]))
            client_train_losses.append(client_train_loss)

        local_time_sec = time.perf_counter() - local_t0

        # 3. Jointly select integer B_n and common computation duration l_c
        # under the paper's quantization-error and wireless-time constraints.
        (
            client_bits,
            weighted_quantization_error,
            planned_compute_time,
            planned_round_time,
        ) = allocate_bits_joint(
            delta_scales=client_delta_scales,
            client_sizes=client_sizes,
            epsilon=epsilon,
            local_epochs=num_epochs,
            min_bits=MIN_QUANTIZATION_BITS,
            max_bits=MAX_QUANTIZATION_BITS,
            solver_maxiter=WIRELESS_SOLVER_MAXITER,
            model_update_dim=model_update_dim,
            client_compute_cycles_per_sample=client_compute_cycles_per_sample,
            channel_power_gains=channel_power_gains,
        )

        # Quantize every floating-point coordinate with the client-wide range.
        client_deltas_list = []
        message_bits_per_client = []
        for cid, raw_delta in enumerate(client_raw_deltas):
            B = client_bits[cid]
            quantization_generator = torch.Generator(device=device.type)
            quantization_generator.manual_seed(
                WIRELESS_RANDOM_SEED * 1000000 + rnd * 100 + cid
            )
            diff_min, diff_max = client_delta_ranges[cid]
            delta_w = {}
            for key in global_params:
                if key in float_update_keys:
                    delta_w[key] = quantize_tensor(
                        raw_delta[key],
                        B,
                        abs_min=diff_min,
                        abs_max=diff_max,
                        generator=quantization_generator,
                    )
                else:
                    delta_w[key] = torch.zeros_like(
                        global_params[key], dtype=torch.float32
                    )
            client_deltas_list.append(delta_w)

            # S_n = d(B_n + 1) + m: B magnitude bits, one sign bit,
            # and m metadata bits for the client-wide min/max range.
            message_bits_per_client.append(
                int(model_update_dim * (B + 1) + QUANTIZATION_METADATA_BITS)
            )
            payload_events.append(
                build_payload_event(
                    rnd,
                    cid,
                    "uplink",
                    "quantized_update",
                    summarize_wireless_payload(
                        raw_delta,
                        delta_w,
                        B,
                        QUANTIZATION_METADATA_BITS,
                        model_update_dim,
                        float_update_keys,
                    ),
                    error_norm=payload_error_norm(
                        raw_delta, delta_w, float_update_keys
                    ),
                )
            )

        wireless_cost = compute_wireless_round_cost(
            message_bits=message_bits_per_client,
            client_sizes=client_sizes,
            local_epochs=num_epochs,
            common_compute_time=planned_compute_time,
            client_compute_cycles_per_sample=client_compute_cycles_per_sample,
            channel_power_gains=channel_power_gains,
        )
        wireless_round_costs.append({"round": rnd, **wireless_cost})
        for cid in range(num_clients):
            wireless_cost_log.append(
                {
                    "round": rnd,
                    "client": cid,
                    "bits": message_bits_per_client[cid],
                    "B": client_bits[cid],
                    "delta_n": client_delta_scales[cid],
                    "samples": client_sizes[cid],
                    "minimum_compute_time_sec": wireless_cost[
                        "minimum_compute_time_sec"
                    ],
                    "compute_time_sec": float(
                        wireless_cost["compute_time_per_client_sec"][cid]
                    ),
                    "effective_cpu_frequency_hz": float(
                        wireless_cost["effective_cpu_frequency_hz"][cid]
                    ),
                    "uplink_rate_bps": float(wireless_cost["uplink_rate_bps"][cid]),
                    "uplink_time_sec": float(
                        wireless_cost["uplink_time_per_client_sec"][cid]
                    ),
                    "compute_energy_joule": float(
                        wireless_cost["compute_energy_per_client_joule"][cid]
                    ),
                    "uplink_energy_joule": float(
                        wireless_cost["uplink_energy_per_client_joule"][cid]
                    ),
                    "total_energy_joule": float(
                        wireless_cost["total_energy_per_client_joule"][cid]
                    ),
                }
            )

        # 4. Server-side Flower/FedAvg aggregation of the compressed updates
        server_t0 = time.perf_counter()
        # using the same num_examples weights returned by each client.
        averaged_delta = fed_avg(client_deltas_list, client_sizes)

        # Update global model using the aggregated differential. Non-floating
        # bookkeeping buffers stay at the server and are not model coordinates.
        updated_global_params = {}
        for key in global_params:
            if key in float_update_keys:
                updated_global_params[key] = global_params[key] + averaged_delta[key]
            else:
                updated_global_params[key] = global_params[key]
        set_params(global_model, updated_global_params)

        # 4. Evaluate global model
        metrics = evaluate(global_model, global_testloader, device, num_classes)
        server_time_sec = time.perf_counter() - server_t0
        train_loss = float(np.mean(client_train_losses))
        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_vram_mb = 0.0

        round_metrics = {
            "round": rnd,
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_macro": metrics["recall_macro"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_micro": metrics["precision_micro"],
            "recall_micro": metrics["recall_micro"],
            "f1_micro": metrics["f1_micro"],
            "worst_class_f1": metrics["worst_class_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "train_loss": train_loss,
            "local_time_sec": round(local_time_sec, 2),
            "server_time_sec": round(server_time_sec, 2),
            "peak_vram_mb": round(float(peak_vram_mb), 2),
        }
        # Keep Wireless-specific round metrics separate for a later table.
        wireless_metrics_log.append(
            {
                "round": rnd,
                "epsilon": epsilon,
                "weighted_quantization_error": weighted_quantization_error,
                "mean_bits_per_coordinate": float(
                    np.average(client_bits, weights=client_sizes)
                ),
                "communication_bits": int(sum(message_bits_per_client)),
                "communication_megabytes": float(
                    sum(message_bits_per_client) / 8.0 / 1e6
                ),
                "optimized_compute_time_sec": float(planned_compute_time),
                "optimized_round_time_sec": float(planned_round_time),
                "compute_time_sec": wireless_cost["compute_time_sec"],
                "communication_time_sec": wireless_cost["communication_time_sec"],
                "wireless_round_time_sec": wireless_cost["round_time_sec"],
                "compute_energy_joule": float(
                    np.sum(wireless_cost["compute_energy_per_client_joule"])
                ),
                "uplink_energy_joule": float(
                    np.sum(wireless_cost["uplink_energy_per_client_joule"])
                ),
                "total_energy_joule": wireless_cost["total_energy_joule"],
                "max_client_energy_joule": float(
                    np.max(wireless_cost["total_energy_per_client_joule"])
                ),
                "energy_constraint_satisfied": wireless_cost[
                    "energy_constraint_satisfied"
                ],
            }
        )
        metrics_log.append(round_metrics)

        # 5. Save the server global model weights every round
        ckpt_path = checkpoint_dir / f"round_{rnd:02d}_wireless_fedavg.pt"
        save_model_checkpoint(global_model.state_dict(), ckpt_path)
        if num_clients in LATEST_CHECKPOINT_ONLY_CLIENT_COUNTS and rnd > 1:
            prev_ckpt = checkpoint_dir / f"round_{rnd - 1:02d}_wireless_fedavg.pt"
            if prev_ckpt.exists():
                try:
                    prev_ckpt.unlink()
                except Exception:
                    pass
        if rnd == num_rounds:
            last_round_state = {
                "round": rnd,
                "client_bits": client_bits,
                "client_delta_ranges": client_delta_ranges,
                "client_delta_scales": client_delta_scales,
                "client_sizes": client_sizes,
                "client_train_losses": client_train_losses,
                "message_bits_per_client": message_bits_per_client,
                "weighted_quantization_error": weighted_quantization_error,
                "planned_compute_time": planned_compute_time,
                "planned_round_time": planned_round_time,
                "client_raw_deltas": client_raw_deltas,
                "client_quantized_deltas": client_deltas_list,
                "averaged_delta": averaged_delta,
            }

        # 6. Print log
        print(
            f"Round {rnd:02d}/{num_rounds} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1_W={metrics['f1_weighted']:.4f} | "
            f"F1_M={metrics['f1_macro']:.4f} | "
            f"Bits={sum(message_bits_per_client) / 1e6:.2f}M | "
            f"Wireless={wireless_cost['round_time_sec']:.4f}s | "
            f"Local={local_time_sec:.1f}s | Server={server_time_sec:.1f}s"
        )
    resource_state = {
        "client_compute_cycles_per_sample": client_compute_cycles_per_sample,
        "client_distances_m": client_distances_m,
        "channel_power_gains": channel_power_gains,
        "wireless_rng_state": wireless_rng.bit_generator.state,
        "model_update_dim": model_update_dim,
        "float_update_keys": float_update_keys,
        "bandwidth_hz": WIRELESS_BANDWIDTH_HZ,
        "noise_spectral_density_w_hz": NOISE_SPECTRAL_DENSITY_W_HZ,
        "client_cpu_frequency_hz": CLIENT_CPU_FREQUENCY_HZ,
        "cpu_energy_coefficient": CPU_ENERGY_COEFFICIENT,
        "client_energy_budget_joule": CLIENT_ENERGY_BUDGET_J,
        "path_loss_exponent": PATH_LOSS_EXPONENT,
        "random_seed": WIRELESS_RANDOM_SEED,
        "quantization_metadata_bits": QUANTIZATION_METADATA_BITS,
    }
    resource_rows = [
        {
            "client": cid,
            "compute_cycles_per_sample": float(client_compute_cycles_per_sample[cid]),
            "distance_m": float(client_distances_m[cid]),
            "channel_power_gain": float(channel_power_gains[cid]),
            "max_cpu_frequency_hz": CLIENT_CPU_FREQUENCY_HZ,
            "energy_budget_joule": CLIENT_ENERGY_BUDGET_J,
            "cpu_energy_coefficient": CPU_ENERGY_COEFFICIENT,
            "bandwidth_hz": WIRELESS_BANDWIDTH_HZ,
            "noise_spectral_density_w_hz": NOISE_SPECTRAL_DENSITY_W_HZ,
            "path_loss_exponent": PATH_LOSS_EXPONENT,
            "random_seed": WIRELESS_RANDOM_SEED,
            "model_update_dim": model_update_dim,
            "quantization_metadata_bits": QUANTIZATION_METADATA_BITS,
        }
        for cid in range(num_clients)
    ]
    return {
        "model": global_model,
        "metrics_log": metrics_log,
        "payload_events": payload_events,
        "wireless_metrics_log": wireless_metrics_log,
        "wireless_cost_log": wireless_cost_log,
        "wireless_round_costs": wireless_round_costs,
        "resource_state": resource_state,
        "last_round_state": last_round_state,
        "extra_tables": {
            "wireless_metrics_per_round.csv": wireless_metrics_log,
            "wireless_cost_per_client.csv": wireless_cost_log,
            "wireless_resources.csv": resource_rows,
        },
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
