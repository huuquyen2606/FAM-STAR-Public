"""Prepared-data serialization and PyTorch dataset helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .preprocessing import PreparedData


def save_prepared_data(
    prepared_data: PreparedData,
    client_indices: list[np.ndarray],
    output_dir: str | Path,
    num_clients: int,
    seed: int,
    batch_size: int = 128,
) -> Path:
    """Save the NumPy/JSON/Pickle artifact layout consumed by current notebooks."""
    save_dir = Path(output_dir)
    clients_dir = save_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_features": len(prepared_data.selected_feature_cols),
        "num_classes": len(prepared_data.label_encoder.classes_),
        "num_clients": num_clients,
        "classes": prepared_data.label_encoder.classes_.tolist(),
        "feature_cols": prepared_data.selected_feature_cols,
        "seed": seed,
        "batch_size": batch_size,
        "val_ratio": 0.1,
    }
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print("metadata.json saved")

    with (save_dir / "preprocessors.pkl").open("wb") as file:
        pickle.dump(prepared_data.preprocessors, file)
    print("preprocessors.pkl saved")

    np.save(save_dir / "X_test.npy", prepared_data.X_test.astype(np.float32))
    np.save(save_dir / "y_test.npy", prepared_data.y_test.astype(np.int64))
    print(f"X_test.npy saved {prepared_data.X_test.shape}")
    print(f"y_test.npy saved {prepared_data.y_test.shape}")

    for client_id, indices in enumerate(client_indices):
        X_client = prepared_data.X_train[indices].astype(np.float32)
        y_client = prepared_data.y_train[indices].astype(np.int64)
        np.save(clients_dir / f"client_{client_id}_X.npy", X_client)
        np.save(clients_dir / f"client_{client_id}_y.npy", y_client)
        print(f"client_{client_id}: {X_client.shape}")

    print(f"Saved data: {save_dir}")
    return save_dir


def choose_torch_device(feature_group_size: int = 21):
    """Check whether the current PyTorch build can run the notebook's LSTM."""
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = nn.LSTM(
            input_size=feature_group_size,
            hidden_size=1,
            batch_first=True,
        ).to("cuda")
        x_probe = torch.randn(1, 2, feature_group_size, device="cuda")
        with torch.no_grad():
            probe(x_probe)
        torch.cuda.synchronize()
        del probe, x_probe
        return torch.device("cuda")
    except Exception as exc:
        print(f"CUDA is visible but unusable for this PyTorch/RNN build: {exc}")
        print(
            "Falling back to CPU. In Kaggle, switch GPU type or install a "
            "PyTorch build matching the accelerator to use CUDA."
        )
        return torch.device("cpu")


def reshape_for_blstm_gru(
    features: np.ndarray,
    group_size: int = 9,
) -> np.ndarray:
    """Convert tabular rows to shorter feature sequences for faster BLSTM-GRU."""
    features = np.asarray(features, dtype=np.float32)
    remainder = features.shape[1] % group_size
    if remainder:
        pad_width = group_size - remainder
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    return features.reshape(features.shape[0], -1, group_size)


def make_client_loaders(
    client_indices,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    feature_group_size: int = 21,
):
    """Build the same in-memory client and test DataLoaders as the notebooks."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    client_trainloaders = {}
    client_items = (
        client_indices.items()
        if hasattr(client_indices, "items")
        else enumerate(client_indices)
    )

    for client_id, index_list in client_items:
        X_client = torch.tensor(
            reshape_for_blstm_gru(
                X_train[index_list],
                feature_group_size,
            ),
            dtype=torch.float32,
        )
        y_client = torch.tensor(y_train[index_list], dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_client, y_client),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        client_trainloaders[client_id] = loader
        print(
            f"Client {client_id:02d}: {len(index_list):>6} samples | "
            f"{len(loader):>4} batches/epoch"
        )

    X_test_tensor = torch.tensor(
        reshape_for_blstm_gru(X_test, feature_group_size),
        dtype=torch.float32,
    )
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=512,
        shuffle=False,
    )
    print(f"\nTest loader: {len(X_test)} samples | {len(test_loader)} batches")
    print(
        "BLSTM-GRU input shape per sample: "
        f"{tuple(X_test_tensor.shape[1:])}"
    )

    return client_trainloaders, test_loader
