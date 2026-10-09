"""Shared data loading, model, and training-state helpers for FL baselines."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
                f"metadata.json under {kaggle_dir}. Set DATA_DIR in the runner to override."
            )
        return kaggle_dir

    return (
        project_root / "data" / "processed" / "CICAndMal2020" / f"{num_clients}clients"
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

    global_testloader = make_test_loader(X_test, y_test, test_batch_size)
    print(
        f"\nGlobal Test Loader: {len(X_test)} samples | "
        f"{len(global_testloader)} batches"
    )
    return client_trainloaders, global_testloader


class HybridBLSTM_GRU(nn.Module):
    """Notebook BLSTM-GRU with the same parameter names and initialization."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        blstm_hidden: int = 300,
        gru_hidden: int = 100,
        dense_hidden: int = 80,
        dropout: float = 0.3,
        num_layers: int = 1,
        flatten_parameters: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.lstm_hidden = int(blstm_hidden)
        self.gru_hidden = int(gru_hidden)
        self.fc_hidden = int(dense_hidden)
        self.dropout_p = float(dropout)
        self.flatten_parameters = flatten_parameters

        self.blstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.ln_blstm = nn.LayerNorm(self.lstm_hidden * 2)
        self.gru = nn.GRU(
            input_size=self.lstm_hidden * 2,
            hidden_size=self.gru_hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.ln_gru = nn.LayerNorm(self.gru_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc1 = nn.Linear(self.gru_hidden, self.fc_hidden)
        self.ln_fc = nn.LayerNorm(self.fc_hidden)
        self.fc_out = nn.Linear(self.fc_hidden, self.num_classes)

    def forward(self, x, return_features: bool = False):
        if self.flatten_parameters:
            self.blstm.flatten_parameters()
            self.gru.flatten_parameters()

        out, _ = self.blstm(x)
        out = self.relu(self.ln_blstm(out[:, -1, :]))
        out, _ = self.gru(out.unsqueeze(1))
        out = self.dropout(self.relu(self.ln_gru(out[:, -1, :])))
        features = self.relu(self.ln_fc(self.fc1(out)))
        logits = self.fc_out(features)
        if return_features:
            return logits, features
        return logits


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_params(model: nn.Module):
    return copy.deepcopy(model.state_dict())


def set_params(model: nn.Module, params) -> None:
    model.load_state_dict(params, strict=True)


def fed_avg(client_params_list, client_sizes):
    """Sample-count averaging with the notebook's add_(alpha=...) ordering."""
    total = sum(client_sizes)
    averaged = {}
    for name, first in client_params_list[0].items():
        if torch.is_floating_point(first):
            averaged[name] = torch.zeros_like(first)
            for params, size in zip(client_params_list, client_sizes):
                averaged[name].add_(
                    params[name].to(dtype=averaged[name].dtype),
                    alpha=size / total,
                )
        else:
            averaged[name] = first.clone()
    return averaged


def make_test_loader(X, y, batch_size: int = 256):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def _load_validation_data(
    data_dir: Path,
    metadata,
    client_X_list,
    client_y_list,
    seed: int,
):
    """Preserve the GKD-VOTE explicit-validation or per-client holdout branch."""
    val_x_path = data_dir / "X_val.npy"
    val_y_path = data_dir / "y_val.npy"
    if val_x_path.is_file() and val_y_path.is_file():
        return np.load(val_x_path), np.load(val_y_path)

    validation_ratio = float(metadata.get("val_ratio", 0.1))
    val_X_parts, val_y_parts = [], []
    for client_id, (X_client, y_client) in enumerate(zip(client_X_list, client_y_list)):
        n_val = min(
            max(int(round(len(y_client) * validation_ratio)), 1),
            len(y_client) - 1,
        )
        if n_val <= 0:
            continue
        indices = np.random.default_rng(seed + client_id).permutation(len(y_client))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        val_X_parts.append(X_client[val_idx])
        val_y_parts.append(y_client[val_idx])
        client_X_list[client_id] = X_client[train_idx]
        client_y_list[client_id] = y_client[train_idx]
    if not val_X_parts:
        raise RuntimeError("Could not create a validation set from client data.")
    return np.concatenate(val_X_parts, axis=0), np.concatenate(val_y_parts, axis=0)


def load_baseline_data(
    num_clients: int,
    batch_size: int,
    test_batch_size: int,
    seed: int,
    project_root: str | Path,
    data_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    validation: bool = False,
):
    """Load shared inputs; only GKD-VOTE requests its notebook validation split."""
    set_seed(seed)
    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    selected_data_dir = resolve_cicandmal_data_dir(
        num_clients,
        data_dir=data_dir,
        project_root=project_root,
    )
    metadata, client_X_list, client_y_list, X_test, y_test = load_cicandmal_data(
        selected_data_dir,
        expected_clients=num_clients,
    )
    print(f"Running on device: {selected_device}")
    print(f"Using data from: {selected_data_dir}")
    print(f"Classes: {metadata['classes']}")
    print(f"Number of features: {metadata['num_features']}")
    print(f"Number of clients: {num_clients}")
    print(f"Number of classes: {metadata['num_classes']}")

    global_valloader = None
    if validation:
        X_val, y_val = _load_validation_data(
            selected_data_dir, metadata, client_X_list, client_y_list, seed
        )
        global_valloader = make_test_loader(X_val, y_val, test_batch_size)
        print(f"Global Validation Data: X_val = {X_val.shape}, y_val = {y_val.shape}")

    client_trainloaders, global_testloader = make_client_loaders(
        client_X_list,
        client_y_list,
        X_test,
        y_test,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
    )
    return {
        "metadata": metadata,
        "classes": metadata["classes"],
        "num_features": int(metadata["num_features"]),
        "num_classes": int(metadata["num_classes"]),
        "num_clients": num_clients,
        "client_X_list": client_X_list,
        "client_y_list": client_y_list,
        "client_trainloaders": client_trainloaders,
        "global_testloader": global_testloader,
        "global_valloader": global_valloader,
        "device": selected_device,
        "data_dir": selected_data_dir,
    }
