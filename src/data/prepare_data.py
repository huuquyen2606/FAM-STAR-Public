"""Command-line entry point for preparing CICAndMal2020 client datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .dataset import save_prepared_data
from .partition import (
    build_client_label_distribution,
    make_noniid_label_split_with_trick,
    plot_client_label_distribution,
)
from .preprocessing import preprocess_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "CICAndMal2020.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess CICAndMal2020 and partition training data by client."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Raw CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (defaults under {DEFAULT_OUTPUT_ROOT}/CICAndMal2020/<N>clients).",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        choices=(10, 20, 50),
        default=10,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-size-target", type=int, default=500)
    parser.add_argument("--max-retries", type=int, default=1000)
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Open the label-distribution chart after saving it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input_path.expanduser().resolve()
    if args.output_dir is None:
        output_dir = (
            DEFAULT_OUTPUT_ROOT
            / "CICAndMal2020"
            / f"{args.num_clients}clients"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {input_path}")

    print(f"Reading dataset: {input_path}")
    df = pd.read_csv(input_path)
    prepared_data = preprocess_dataframe(
        df,
        seed=args.seed,
        test_size=args.test_size,
    )

    label_list = prepared_data.label_encoder.classes_.tolist()
    num_classes = len(label_list)
    print(f"Classes: {num_classes}")
    print(f"Class labels: {label_list}")
    print(f"Train raw: {prepared_data.X_train_raw_shape}")
    print(f"Test raw: {prepared_data.X_test_raw_shape}")
    print(f"Features after selection: {len(prepared_data.selected_feature_cols)}")
    print(f"X_train: {prepared_data.X_train.shape}")
    print(f"X_test: {prepared_data.X_test.shape}")

    client_indices = make_noniid_label_split_with_trick(
        prepared_data.y_train,
        num_clients=args.num_clients,
        seed=args.seed,
        min_size_target=args.min_size_target,
        max_retries=args.max_retries,
    )
    for client_id, indices in enumerate(client_indices):
        print(f"Client {client_id}: {len(indices)} samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    client_label_distribution_df = build_client_label_distribution(
        client_indices,
        prepared_data.y_train,
        prepared_data.label_encoder,
    )
    distribution_path = output_dir / "client_label_distribution.csv"
    client_label_distribution_df.to_csv(distribution_path, index=False)
    print(f"Saved to: {distribution_path}")

    chart_path = output_dir / "label_distribution_bubble_modern.png"
    plot_client_label_distribution(
        client_indices,
        prepared_data.y_train,
        label_list,
        chart_path,
        show=args.show_plot,
    )
    print(f"Saved chart to: {chart_path}")

    save_prepared_data(
        prepared_data,
        client_indices,
        output_dir,
        num_clients=args.num_clients,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
