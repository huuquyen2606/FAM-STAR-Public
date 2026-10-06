"""Load the preprocessed CICAndMal2020 arrays used by FL baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

KAGGLE_DATASET_SLUGS = {
    10: "cicandmal2020-10clients",
    20: "cicandmal2020-20clients",
    50: "cicandmal2020-50clients",
}


def _has_client_count(data_dir: Path, expected_clients: int) -> bool:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return int(metadata.get("num_clients", -1)) == expected_clients
    except (OSError, TypeError, ValueError):
        return False


def resolve_cicandmal_data_dir(
    num_clients: int,
    data_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve an explicit path, one fixed Kaggle dataset, or local processed data."""
    project_root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    if data_dir is not None:
        data_dir = Path(data_dir).expanduser()
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
        return data_dir.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        dataset_slug = KAGGLE_DATASET_SLUGS[num_clients]
        kaggle_dir = kaggle_input / "datasets" / "zhacutis1tg" / dataset_slug
        if not _has_client_count(kaggle_dir, num_clients):
            dataset_url = f"https://www.kaggle.com/datasets/zhacutis1tg/{dataset_slug}"
            raise FileNotFoundError(
                f"Attach {dataset_url} as a Kaggle Input; expected "
                f"metadata.json under {kaggle_dir}. Use --data-dir to override."
            )
        return kaggle_dir

    return (
        project_root
        / "data"
        / "processed"
        / "CICAndMal2020"
        / f"{num_clients}clients"
    )


def load_cicandmal_data(data_dir: str | Path, expected_clients: int):
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    num_clients = int(metadata["num_clients"])
    if num_clients != expected_clients:
        raise ValueError(
            f"Expected {expected_clients} clients; metadata has {num_clients}."
        )

    client_X_list = []
    client_y_list = []
    clients_dir = data_dir / "clients"
    for client_id in range(num_clients):
        X_client = np.load(clients_dir / f"client_{client_id}_X.npy")
        y_client = np.load(clients_dir / f"client_{client_id}_y.npy")
        client_X_list.append(X_client)
        client_y_list.append(y_client)

    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    return metadata, client_X_list, client_y_list, X_test, y_test


def make_client_loaders(
    client_X_list,
    client_y_list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    test_batch_size: int = 256,
):
    """Create the baseline input shape ``(N, 1, num_features)``."""
    client_trainloaders = {}
    for client_id, (X_client, y_client) in enumerate(zip(client_X_list, client_y_list)):
        X_client_tensor = torch.tensor(X_client, dtype=torch.float32).unsqueeze(1)
        y_client_tensor = torch.tensor(y_client, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_client_tensor, y_client_tensor),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        client_trainloaders[client_id] = loader
        print(
            f"Client {client_id:02d}: {len(y_client)} samples | "
            f"{len(loader)} batches/epoch"
        )

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    global_testloader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=test_batch_size,
        shuffle=False,
    )
    print(
        f"\nGlobal Test Loader: {len(X_test)} samples | "
        f"{len(global_testloader)} batches"
    )
    return client_trainloaders, global_testloader
