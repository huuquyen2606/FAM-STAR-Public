"""Command-line runner for FAM-STAR."""

from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse

import torch

from src.main import FrameworkConfig, run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run FAM-STAR on prepared CICAndMal2020 clients."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Prepared dataset directory containing metadata.json, "
            "X_test.npy, y_test.npy, and clients/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="fam_star_results",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        
    config = FrameworkConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=torch.device(args.device),
    )
    run_training_pipeline(config)


if __name__ == "__main__":
    main()
