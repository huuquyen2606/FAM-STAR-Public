# FAM-STAR

**Federated Android Malware Classification with Sparse Topology Adaptation and Role-Specific Aggregation**

FAM-STAR is a federated learning framework for Android malware classification under fragmented label support, non-IID client data, and communication constraints. It combines class-aware local learning, adaptive sparse model connectivity, role-specific aggregation, and compact exchange of model updates and class prototypes.

This repository contains the modular FAM-STAR implementation and eight comparison baselines for **10, 20, and 50 clients**. Clients are logical participants in a synchronous federated simulation, not separate Android devices.

## Method Overview

- **Class-aware supervision:** effective-number class balancing and restricted softmax preserve the global label space while down-scaling logits for locally absent classes.
- **Masked FedProx:** local proximal regularization respects the current sparse mask.
- **Prototype-guided learning:** global class prototypes support embedding alignment and an auxiliary classifier-head cross-entropy objective.
- **Dynamic sparse topology:** prune/regrow proposals use classification-gradient evidence; the classification head remains dense.
- **Role-specific aggregation:** shared representation parameters use client-level class-diversity evidence, while classifier rows use class-specific support.
- **Prototype fusion:** local class prototypes are aggregated using support and embedding dispersion.
- **Compact uplink:** sparse deltas and prototypes use INT8 quantization, bit-packed maps, and client-side error feedback.

Each round activates the previously agreed mask, trains participating clients, aggregates their updates and prototypes, proposes the next mask, evaluates the global model, and saves results. A newly agreed mask becomes active in the **next** round. Only the CB-RS classification gradient supplies pruning/regrowth evidence.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- .gitignore
|-- data/
|   |-- raw/
|   |   `-- CICAndMal2020.csv
|   `-- processed/
|       `-- CICAndMal2020/
|           |-- 10clients/
|           |-- 20clients/
|           `-- 50clients/
|-- checkpoints/
|   `-- .gitkeep
|-- experiments/
|   |-- .gitkeep
|   |-- fedavg/
|   |-- fedprox/
|   |-- jopeq/
|   |-- wireless/
|   |-- feddst/
|   |-- zeroshot/
|   |-- fedgpd/
|   `-- fedgkd_vote/
`-- src/
    |-- main.py
    |-- train.py
    |-- evaluate.py
    |-- scripts/
    |   `-- run_fam_star.py
    |-- data/
    |   |-- prepare_data.py
    |   |-- preprocessing.py
    |   |-- partition.py
    |   `-- dataset.py
    |-- models/
    |   `-- hybrid_blstm_gru.py
    |-- fam_star/
    |   |-- prototype.py
    |   |-- topology.py
    |   |-- aggregation.py
    |   `-- compression.py
    |-- baselines/
    |   |-- common/
    |   |   |-- data.py
    |   |   |-- metrics.py
    |   |   `-- results.py
    |   |-- fedavg.py
    |   |-- fedprox.py
    |   |-- jopeq.py
    |   |-- wireless.py
    |   |-- feddst.py
    |   |-- zeroshot.py
    |   |-- fedgpd.py
    |   `-- fedgkd_vote.py
    `-- utils/
        |-- metrics.py
        |-- seed.py
        `-- io.py
```

Package `__init__.py` files and generated `__pycache__/` directories are omitted. Data and experiment artifacts must be obtained or generated separately when absent from a checkout. Each baseline method directory contains `10clients/`, `20clients/`, and `50clients/` result slots; an empty slot is not a completed experiment.

### File and Folder Responsibilities

| Path | Responsibility |
|---|---|
| `data/raw/` | Input CSV used by the preparation pipeline. |
| `data/processed/` | Prepared client arrays, shared test arrays, metadata, and preprocessing artifacts. |
| `checkpoints/` | Root placeholder; current runners save checkpoints inside their selected result directory, not here by default. |
| `experiments/` | Locally organized baseline outputs; FAM-STAR can also write here through `--output-dir`. |
| `requirements.txt` / `.gitignore` | Pinned dependencies / exclusions for caches, environments, checkpoints, and generated baseline outputs. |
| [src/main.py](src/main.py) | `FrameworkConfig` and the end-to-end FAM-STAR training pipeline. |
| [src/train.py](src/train.py) | Local CB-RS, masked FedProx, prototype objectives, and topology-evidence collection. |
| [src/evaluate.py](src/evaluate.py) | Global-model evaluation on the shared test set. |
| [src/scripts/run_fam_star.py](src/scripts/run_fam_star.py) | FAM-STAR command-line entry point. |
| [src/data/prepare_data.py](src/data/prepare_data.py) | CSV-to-prepared-partition command-line entry point. |
| [src/data/preprocessing.py](src/data/preprocessing.py) | Label encoding, train/test splitting, imputation, feature selection, and standardization. |
| [src/data/partition.py](src/data/partition.py) | Fragmented-label non-IID partitioning and client-label distribution exports. |
| [src/data/dataset.py](src/data/dataset.py) | Prepared-data serialization/loading, data loaders, and client class statistics. |
| [src/models/hybrid_blstm_gru.py](src/models/hybrid_blstm_gru.py) | FAM-STAR's BLSTM-GRU classifier and embedding output. |
| [src/fam_star/prototype.py](src/fam_star/prototype.py) | Class-prototype extraction, quantization, dispersion, and server fusion. |
| [src/fam_star/topology.py](src/fam_star/topology.py) | Sparse model masks, pruning/regrowth, and server mask consensus; not a client-network graph. |
| [src/fam_star/aggregation.py](src/fam_star/aggregation.py) | Separate aggregation weights for shared tensors and class-specific classifier rows. |
| [src/fam_star/compression.py](src/fam_star/compression.py) | Sparse INT8 delta exchange, error feedback, bit-packed maps, and payload accounting. |
| `src/baselines/*.py` | Independent method-specific helpers, editable settings, `train_federated`, and `main`. |
| [src/baselines/common/data.py](src/baselines/common/data.py) | Baseline model, seeding, data/validation loading, and model-state/averaging helpers. |
| [src/baselines/common/metrics.py](src/baselines/common/metrics.py) | Baseline predictive metrics, with optional loss and MCC. |
| [src/baselines/common/results.py](src/baselines/common/results.py) | Baseline output paths, payload events, checkpoint saving, CSVs, reports, and figures. |
| [src/utils/metrics.py](src/utils/metrics.py) | FAM-STAR metric/communication exports, classification report, and figures. |
| [src/utils/seed.py](src/utils/seed.py) | FAM-STAR seeding utilities. |
| [src/utils/io.py](src/utils/io.py) | FAM-STAR round-state checkpoint serialization; no resume entry point is currently provided. |

## Installation

The recorded research environment used Python **3.12.13**, PyTorch **2.10.0+cu128**, and a Kaggle NVIDIA Tesla T4 on `cuda:0`. Although that runtime exposed two GPUs, training used one. Package versions are pinned in `requirements.txt`; the installed CUDA build depends on the PyTorch installation.

```bash
git clone https://github.com/huuquyen2606/FAM-STAR-Public.git
cd FAM-STAR-Public
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the pinned dependencies with `python -m pip install -r requirements.txt`.

The implementation uses PyTorch, NumPy, pandas, scikit-learn, matplotlib, seaborn, and SciPy. Wireless uses SciPy for its Lambert-W resource solver. Runners select `cuda:0` when available and otherwise use CPU; full CPU experiments can be substantially slower.

## Dataset Preparation

Dataset: [CCCS-CIC-AndMal-2020](https://www.unb.ca/cic/datasets/andmal2020.html). The study uses 53,439 records, 126 selected features, and 14 labels, with a stratified 80/20 split: 42,751 training records and 10,688 held-out test records.

For paper reproduction, use the fixed prepared partitions rather than creating another random split. The [full research archive](https://drive.google.com/drive/folders/1s52jWVlChoOBYSDoP3yDJZMludPYOnM0) contains `01_Code/`, `02_Dataset/`, `03_Data_Preprocessing/`, and `04_Results/`.

| Clients | Local prepared directory | Kaggle dataset |
|---:|---|---|
| 10 | `data/processed/CICAndMal2020/10clients/` | [10-client partition](https://www.kaggle.com/datasets/zhacutis1tg/cicandmal2020-10clients) |
| 20 | `data/processed/CICAndMal2020/20clients/` | [20-client partition](https://www.kaggle.com/datasets/zhacutis1tg/cicandmal2020-20clients) |
| 50 | `data/processed/CICAndMal2020/50clients/` | [50-client partition](https://www.kaggle.com/datasets/zhacutis1tg/cicandmal2020-50clients) |

Each prepared directory has this layout:

```text
<N>clients/
|-- metadata.json
|-- preprocessors.pkl
|-- X_test.npy
|-- y_test.npy
|-- client_label_distribution.csv
|-- label_distribution_bubble_modern.png
`-- clients/
    |-- client_0_X.npy
    |-- client_0_y.npy
    `-- ... client_<N-1>_X.npy and client_<N-1>_y.npy
```

Training loads `metadata.json`, test arrays, and client arrays. Metadata supplies `classes`, `num_classes`, `num_features`, and `num_clients`; loaders validate the requested client count. Preprocessor and distribution files document preparation and are not required by the training loaders.

Baseline data resolution uses an explicit `DATA_DIR` first, the fixed `/kaggle/input/datasets/zhacutis1tg/cicandmal2020-<N>clients` mount on Kaggle, or the local prepared directory above. Attach the matching Kaggle dataset, or set `DATA_DIR` to its actual mount path. FAM-STAR requires an explicit `--data-dir`.

### Optional: Generate Partitions from the CSV

Place the study CSV at `data/raw/CICAndMal2020.csv`. To generate a 10-client partition (use `20` or `50` for the other sizes), run:

```bash
python -m src.data.prepare_data --num-clients 10
```

**This command overwrites artifacts in the selected output directory.** Do not rerun it over archived partitions when reproducing paper results; use `--output-dir` for a separate destination.

Preprocessing replaces infinities with missing values, removes identifier/label columns, performs the stratified split, then fits mean imputation, zero-variance filtering, and `StandardScaler` on training data only. Partitioning uses fragmented label support, not a Dirichlet sampler. The default minimum client size of 500 is a retry target, not a guarantee when the best available split is returned.

## Configuration

### Shared Model and Training Defaults

| Component | Setting |
|---|---|
| Base classifier | BiLSTM (300) -> GRU (100) -> dense (80) -> global-class output; dropout 0.3 |
| Optimizer / learning rate | Adamax / 0.002 |
| Rounds / local epochs | 50 / 5 |
| Training / test batch size | 128 / 256 |
| Seed / participation | 42 / all clients each round |

Tabular samples are supplied as `(batch, 1, num_features)`, not multi-step sequences. ZeroShot structurally reduces the base model before federated training.

### FAM-STAR Settings

Algorithm settings live in `FrameworkConfig` in `src/main.py`:

| Fields | Defaults |
|---|---|
| `fedprox_mu` | 0.01 |
| `cb_beta`, `missing_class_scale` | 0.90, 0.10; present-class logits keep scale 1.0 |
| `aggregation_gamma` | 0.50 |
| `total_sparsity`, `prune_fraction` | 0.50, 0.05 |
| `prototype_lambda` | 0.05 for both alignment and prototype-head CE |
| `prototype_eps`, `int8_eps` | `1e-8`, `1e-8` |

Classification-gradient EMA decay is 0.90 in `src/train.py`. Sparsity applies to designated shared tensors, not the classification head. Prototype auxiliary terms are inactive in round 1 until global prototypes have been aggregated.

### Baseline Settings

All eight baseline runners use the same code layout. Settings are constants near the top of each method file, **not command-line arguments**.

| Method | File | Evaluated setting |
|---|---|---|
| FedAvg | `fedavg.py` | Sample-count-weighted model averaging |
| FedProx | `fedprox.py` | `FEDPROX_MU = 0.01` for every client count |
| JoPEQ | `jopeq.py` | `JOPEQ_EPSILON = 3.0`, nominal `JOPEQ_B = 4.0` (`R = 4`) |
| Wireless Quantized FL | `wireless.py` | `WIRELESS_EPSILON = 0.01` for every client count |
| FedDST | `feddst.py` | `SPARSITY = 0.50`, `MASK_UPDATE_INTERVAL = 15` |
| Zero-shot pruning | `zeroshot.py` | `PRUNING_RATE = 0.50`; uniform server averaging |
| FedGKD-VOTE | `fedgkd_vote.py` | `FEDGKD_M = 5`, `FEDGKD_GAMMA = 0.5`, `TEMPERATURE = 2.0`, `FEDGKD_BETA = 1.0 / FEDGKD_M` |
| FedGPD | `fedgpd.py` | `TEMPERATURE = 2.0`, `FEDGPD_LAMBDA = 0.05` |

FedProx, FedDST, and Wireless use the selected settings above rather than differing values in the original notebooks. Method-specific state and diagnostics remain separate from the three shared baseline modules.

## Running Experiments

Run all commands from the repository root.

### FAM-STAR

Unlike the baselines, the existing FAM-STAR entry point accepts CLI flags:

```bash
python -m src.scripts.run_fam_star --data-dir data/processed/CICAndMal2020/10clients --num-clients 10 --output-dir experiments/fam_star/10clients
python -m src.scripts.run_fam_star --data-dir data/processed/CICAndMal2020/20clients --num-clients 20 --output-dir experiments/fam_star/20clients
python -m src.scripts.run_fam_star --data-dir data/processed/CICAndMal2020/50clients --num-clients 50 --output-dir experiments/fam_star/50clients
```

These commands use the default training settings. Available overrides are `--num-rounds`, `--num-epochs`, `--batch-size`, `--learning-rate`, and `--device`; run `python -m src.scripts.run_fam_star --help` for usage. Without `--output-dir`, the current default is `fam_star_results/`, not `experiments/`.

On Kaggle, provide the attached dataset path through `--data-dir` and select an output under `/kaggle/working/`, for example `/kaggle/working/fam_star/10clients`.

### Baselines

Set `NUM_CLIENTS = 10`, `20`, or `50` in the selected file, then run its module:

```bash
python -m src.baselines.fedavg
python -m src.baselines.fedprox
python -m src.baselines.jopeq
python -m src.baselines.wireless
python -m src.baselines.feddst
python -m src.baselines.zeroshot
python -m src.baselines.fedgpd
python -m src.baselines.fedgkd_vote
```

Edit `NUM_ROUNDS`, `NUM_EPOCHS`, and other constants in the same file when needed. `DATA_DIR = None`, `OUTPUT_DIR = None`, and `DEVICE = None` select automatic data resolution, the default result location, and CUDA/CPU selection. Explicit overrides also belong in that method file; do not append baseline flags to the command.

## Results and Checkpoints

Baseline outputs go directly to `experiments/<method>/<N>clients/` locally or `/kaggle/working/<method>/<N>clients/` on Kaggle. For example, FedAvg's three result folders are `experiments/fedavg/10clients/`, `20clients/`, and `50clients/`. There is no repeated method/client nesting, copied `src/` tree, or automatic results ZIP.

FAM-STAR writes to the selected `--output-dir`, with separate `metrics/`, `figures/`, and `checkpoints/` subdirectories. Paths in this table are relative to the corresponding run directory:

| Output | Baselines | FAM-STAR | Contents |
|---|---|---|---|
| Round metrics | `metrics_per_round.csv` | `metrics/metrics_per_round.csv` | Predictive metrics, training loss, execution time, and peak CUDA memory |
| Payload events | `payload_events.csv` | `metrics/payload_events.csv` | Client/direction payload accounting |
| Communication totals | `communication_rounds.csv` | `metrics/communication_rounds.csv` | Round upload/download totals and cumulative communication |
| Per-class metrics | `per_class_metrics.csv` | `metrics/per_class_metrics.csv` | Class support, precision, recall, and F1 |
| Classification report | `classification_report.txt` for FedDST/ZeroShot | `metrics/classification_report.txt` | Text classification report |
| Convergence figure | `metrics_per_round.png` | `figures/metrics_per_round.png` | Accuracy and F1 trajectories |
| Confusion matrix | `confusion_matrix.png` | `figures/confusion_matrix.png` | Final-model confusion matrix |
| Checkpoints | `checkpoints/` | `checkpoints/` | Method-specific model/state files |

Baseline exporters retain additional method-specific metric and payload columns. Wireless also writes `wireless_metrics_per_round.csv`, `wireless_cost_per_client.csv`, and `wireless_resources.csv`.

- FedAvg retains only its latest round checkpoint. Wireless does the same for 50 clients but retains all rounds for 10/20 clients. Other baselines retain round checkpoints.
- FedDST saves auxiliary masks and layer-sparsity state; FedGPD saves auxiliary global prototypes and keeps its source zero-prototype regularization in round 1.
- FedGKD-VOTE uses supplied validation arrays or a deterministic client holdout, keeps frozen historical teachers, and saves `best_model_fedgkd_vote.pt` by validation Macro-F1. Final figures use the last-round model, not that best checkpoint.
- FAM-STAR saves `round_<NN>_famstar.pt` with model parameters, current/pending masks, client residuals, global prototypes, round metrics, and RNG states. Saved state does not imply that automatic resume is implemented.

Runners print to the console; they do not automatically create `.log` files. If retaining a console/notebook log, use `<method>-<N>clients.log` without a project-name prefix. Transfer result files separately from any Kaggle workspace source copy.

## Reproducibility and Scope

Use the archived prepared partitions, seed 42, and the stated configuration to reproduce the study. Compare saved CSVs and figures with `04_Results/` in the research archive; empty result folders do not substitute for completed runs.

- Predictive evaluation uses the shared held-out test set. Macro-F1 is the primary metric; exports also include accuracy, balanced accuracy, macro/micro/weighted precision, recall and F1, and worst-class F1.
- Communication is application-level accounting, not measured network traffic or transport overhead. FAM-STAR and dense baseline packages include serialization bytes; JoPEQ uses actual byte encoding/decoding.
- JoPEQ's nominal 4-bit setting has 17 symbols and therefore uses 5 packed bits per symbol. Wireless reports an analytical uplink bit budget and a serialized dense downlink, not an implemented packed uplink codec.
- FedGKD-VOTE preserves notebook accounting differences: only the 10-client variant records additional historical-teacher downlinks. Account for this when comparing communication totals.
- ZeroShot's 50% pruning refers to structural hidden-dimension reduction, not exactly 50% fewer total parameters or a 50% sparse mask.
- A shortened run checks execution only; one round cannot exercise FedDST's round-15 rewiring or historical-teacher distillation. Restore experimental settings before full runs.
- The reported study uses one recorded seed and a single-GPU logical simulation. Different partitions, libraries, CUDA/cuDNN versions, or hardware can change results.
- FAM-STAR assumes honest participants. It does not implement poisoning/backdoor defenses, Byzantine robustness, secure aggregation, or a formal differential-privacy guarantee.

## License

Source code is licensed under [Apache License 2.0](LICENSE). Dataset access, use, and redistribution remain subject to the dataset provider's terms and applicable institutional requirements.
