"""Prepared CICAndMal2020 data loader for FAM-STAR."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


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
