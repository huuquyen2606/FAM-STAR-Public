"""Prepared CICAndMal2020 dataset helpers, loader, and serialization for FAM-STAR."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from .preprocessing import PreparedData


# =====================================================================
# FAM-STAR Dataset Loader & PyTorch Helpers
# =====================================================================

@dataclass
class DatasetBundle:
    classes: list[str]
    num_classes: int
    num_features: int
    num_clients: int
    client_X: list[np.ndarray]
    client_y: list[np.ndarray]
    X_test: np.ndarray
    y_test: np.ndarray
    client_trainloaders: dict[int, DataLoader]
    client_proto_loaders: dict[int, DataLoader]
    global_testloader: DataLoader


def _make_loaders(
    client_X: list[np.ndarray],
    client_y: list[np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    prototype_batch_size: int = 256,
    test_batch_size: int = 256,
) -> tuple[dict[int, DataLoader], dict[int, DataLoader], DataLoader]:
    train_loaders: dict[int, DataLoader] = {}
    proto_loaders: dict[int, DataLoader] = {}

    for cid, (X_c, y_c) in enumerate(zip(client_X, client_y)):
        X_c_t = torch.tensor(X_c, dtype=torch.float32).unsqueeze(1)
        y_c_t = torch.tensor(y_c, dtype=torch.long)
        dataset = TensorDataset(X_c_t, y_c_t)

        train_loaders[cid] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        proto_loaders[cid] = DataLoader(
            dataset,
            batch_size=prototype_batch_size,
            shuffle=False,
            drop_last=False,
        )

    X_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=test_batch_size,
        shuffle=False,
    )
    return train_loaders, proto_loaders, test_loader


def load_prepared_dataset(
    data_dir: str,
    expected_num_clients: int,
    batch_size: int = 128,
) -> DatasetBundle:
    """Load the exact prepared-data layout expected by the original notebook.

    Expected layout:
        data_dir/
          metadata.json
          X_test.npy
          y_test.npy
          clients/
            client_0_X.npy
            client_0_y.npy
            ...
    """
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    classes = list(metadata["classes"])
    num_classes = int(metadata["num_classes"])
    num_features = int(metadata["num_features"])
    num_clients = int(metadata["num_clients"])

    if num_clients != int(expected_num_clients):
        raise ValueError(
            f"Requested {expected_num_clients} clients but metadata reports {num_clients}."
        )

    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    client_X: list[np.ndarray] = []
    client_y: list[np.ndarray] = []
    for cid in range(num_clients):
        client_X.append(
            np.load(os.path.join(data_dir, "clients", f"client_{cid}_X.npy"))
        )
        client_y.append(
            np.load(os.path.join(data_dir, "clients", f"client_{cid}_y.npy"))
        )

    train_loaders, proto_loaders, test_loader = _make_loaders(
        client_X,
        client_y,
        X_test,
        y_test,
        batch_size=batch_size,
    )

    return DatasetBundle(
        classes=classes,
        num_classes=num_classes,
        num_features=num_features,
        num_clients=num_clients,
        client_X=client_X,
        client_y=client_y,
        X_test=X_test,
        y_test=y_test,
        client_trainloaders=train_loaders,
        client_proto_loaders=proto_loaders,
        global_testloader=test_loader,
    )


def compute_client_class_statistics(
    client_y: list[np.ndarray],
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    client_class_counts = np.stack(
        [
            np.bincount(
                np.asarray(y, dtype=np.int64).reshape(-1),
                minlength=int(num_classes),
            ).astype(np.float64)
            for y in client_y
        ],
        axis=0,
    )
    client_sample_counts = np.asarray(
        [len(y) for y in client_y],
        dtype=np.float64,
    )
    sample_weights = client_sample_counts / client_sample_counts.sum()
    global_class_counts = client_class_counts.sum(axis=0)
    return (
        client_class_counts,
        client_sample_counts,
        sample_weights,
        global_class_counts,
    )


# =====================================================================
# Data Preparation & Artifact Serialization
# =====================================================================

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

    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    np.save(save_dir / "X_test.npy", prepared_data.X_test.astype(np.float32))
    np.save(save_dir / "y_test.npy", prepared_data.y_test.astype(np.int64))

    for client_id, indices in enumerate(client_indices):
        X_client = prepared_data.X_train[indices].astype(np.float32)
        y_client = prepared_data.y_train[indices].astype(np.int64)
        np.save(clients_dir / f"client_{client_id}_X.npy", X_client)
        np.save(clients_dir / f"client_{client_id}_y.npy", y_client)

    preprocessor_bundle = {
        "imputer": prepared_data.imputer,
        "selector": prepared_data.selector,
        "scaler": prepared_data.scaler,
        "label_encoder": prepared_data.label_encoder,
    }
    with (save_dir / "preprocessors.pkl").open("wb") as f:
        pickle.dump(preprocessor_bundle, f)

    return save_dir
