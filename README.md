# FAM-STAR

**Federated Android Malware Classification with Sparse Topology Adaptation and Role-Specific Aggregation**

FAM-STAR is a federated learning framework for Android malware classification under **fragmented label support**, **non-IID client data**, and **communication constraints**. The method jointly adapts a sparse shared representation and aggregates client updates according to the role of each parameter: shared representation parameters use client-level class-diversity evidence, while classifier rows use class-specific support. Cross-client prototypes and compact sparse exchange complement the training protocol.

This repository provides the reusable Python implementation of FAM-STAR used for the experiments with **10, 20, and 50 clients**. A single source tree is used for all federation sizes; the client count and prepared data directory are selected at run time.

## Main Components

FAM-STAR combines the following mechanisms:

- **Class-aware local supervision (CB-RS):** effective-number class balancing over locally observed classes while preserving the global output space and down-scaling logits for locally absent classes.
- **Masked FedProx regularization:** constrains local drift while respecting the current sparse topology.
- **Prototype-guided local learning:** global class prototypes provide embedding alignment and an auxiliary classifier-head cross-entropy objective.
- **Class-aware dynamic sparse topology adaptation:** active connections are pruned using classification-derived importance and inactive connections are regrown from gradient-EMA evidence under a fixed sparsity budget.
- **Role-specific server aggregation:** shared representation parameters and classifier rows use different class-evidence-aware aggregation weights.
- **Reliability-weighted prototype aggregation:** class prototypes are fused using local support and embedding dispersion.
- **Compact communication:** sparse model deltas and prototypes are quantized to INT8; support maps and topology masks are bit-packed; error feedback retains local compression residuals.

## Reported Experimental Setting

The accompanying study evaluates FAM-STAR on a processed subset of **CCCS-CIC-AndMal-2020** with:

- **53,439** Android application records.
- **126** input features.
- **14** labels.
- Fixed **80/20** train/test split:
  - 42,751 training records.
  - 10,688 held-out test records.
- Federation sizes: **10, 20, and 50 clients**.
- **50 communication rounds**.
- Full client participation in every round.
- Base random seed: **42**.
- Simulation on a **single Kaggle-hosted NVIDIA Tesla T4 GPU**; clients are logical participants rather than physical devices.

For each federation size, all compared methods use the same client partition and the same held-out test set.

## Model and Training Configuration

The implementation uses the common classifier and FAM-STAR settings from the reported experiments.

| Component | Setting |
|---|---|
| Local model | Bidirectional LSTM -> GRU -> fully connected classifier |
| BiLSTM hidden size | 300 |
| GRU hidden size | 100 |
| Dense hidden size | 80 |
| Dropout | 0.3 |
| Optimizer | Adamax |
| Learning rate | 0.002 |
| Local batch size | 128 |
| Local epochs / round | 5 |
| Test batch size | 256 |
| Communication rounds | 50 |
| Seed | 42 |
| FedProx coefficient `mu` | 0.01 |
| Effective-number coefficient `beta_CB` | 0.90 |
| Missing-class logit scale `alpha` | 0.10 |
| Role-specific aggregation `gamma` | 0.50 |
| Target sparsity | 50% |
| Prune fraction `q` | 0.05 |
| Gradient-EMA decay `delta` | 0.90 |
| Prototype alignment coefficient | 0.05 |
| Prototype-head CE coefficient | 0.05 |
| Prototype epsilon | `1e-8` |
| INT8 quantization epsilon | `1e-8` |

The classification head remains dense. Sparsity is applied only to designated shared tensors.

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
`-- src/
    |-- __init__.py
    |-- main.py
    |-- train.py
    |-- evaluate.py
    |-- scripts/
    |   |-- __init__.py
    |   `-- run_fam_star.py
    |-- data/
    |   |-- __init__.py
    |   `-- dataset.py
    |-- models/
    |   |-- __init__.py
    |   `-- hybrid_blstm_gru.py
    |-- fam_star/
    |   |-- __init__.py
    |   |-- aggregation.py
    |   |-- compression.py
    |   |-- prototype.py
    |   `-- topology.py
    `-- utils/
        |-- __init__.py
        |-- io.py
        |-- metrics.py
        `-- seed.py
```

### Source Responsibilities

- `src/main.py` — end-to-end synchronous FAM-STAR training pipeline and `FrameworkConfig`.
- `src/train.py` — local CB-RS objective, masked FedProx, prototype alignment, prototype-head CE, and topology evidence collection.
- `src/evaluate.py` — shared-test-set evaluation and predictive metrics.
- `src/data/dataset.py` — prepared client-partition and test-set loader.
- `src/models/hybrid_blstm_gru.py` — the BLSTM-GRU classifier.
- `src/fam_star/aggregation.py` — class-evidence-aware, role-specific model aggregation.
- `src/fam_star/topology.py` — sparse-mask initialization, pruning/regrowth, and server topology consensus.
- `src/fam_star/prototype.py` — local prototype extraction, quantization, dispersion, and server prototype fusion.
- `src/fam_star/compression.py` — sparse INT8 delta exchange, error feedback, bit-packed maps, and communication accounting.
- `src/utils/io.py` — per-round checkpoint serialization.
- `src/utils/metrics.py` — metrics, communication tables, reports, and figures.
- `src/utils/seed.py` — deterministic seeding utilities.
- `src/scripts/run_fam_star.py` — command-line entry point.

## Full Reproducibility Archive

The complete research archive, including prepared data, preprocessing artifacts, trained checkpoints, metrics, and figures, is available here:

**FAM-STAR Full Source:**  
https://drive.google.com/drive/folders/1s52jWVlChoOBYSDoP3yDJZMludPYOnM0

The archive is organized into:

```text
FAM-STAR-Full Source/
|-- 01_Code/
|-- 02_Dataset/
|-- 03_Data_Preprocessing/
`-- 04_Results/
```

For reproducing the reported numbers, use the prepared client partitions from the archive rather than generating a new random partition.

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/huuquyen2606/FAM-STAR-Public.git
cd FAM-STAR-Public

python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Current core dependencies are:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
```

### Verified Reproduction Environment

The refactored FAM-STAR implementation was validated on Kaggle and reproduced the reported predictive metrics and communication cost.

| Component | Verified environment |
|---|---|
| Python | `3.12.13` |
| PyTorch | `2.10.0+cu128` |
| CUDA reported by PyTorch | `12.8` |
| NumPy | `2.4.6` |
| pandas | `2.3.3` |
| matplotlib | `3.10.0` |
| SciPy | `1.16.3` |
| scikit-learn | `1.6.1` |
| seaborn | `0.13.2` |
| GPU runtime | Kaggle `GPU T4 x2` |
| GPU used by the experiment | NVIDIA Tesla T4 (`cuda:0`) |

The Kaggle runtime exposed two T4 GPUs, but the FAM-STAR training pipeline used only the primary device (`cuda:0`). Therefore, the reported experiment is a single-GPU simulation, consistent with the experimental description in the manuscript.

## Hardware and Device Selection

The runner accepts an explicit device through `--device`.

When CUDA is available, the default is:

```text
cuda:0
```

Otherwise, the implementation falls back to:

```text
cpu
```

A CPU run is supported by the code path but can be substantially slower for the full 50-round experiment. The reported experiments used one NVIDIA Tesla T4 GPU.

On Kaggle, GPU availability can be checked with:

```python
import torch

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## Prepared Dataset Format

The training runner expects one prepared directory for each federation size.

Example for 10 clients:

```text
/path/to/cicandmal2020-10clients/
|-- metadata.json
|-- X_test.npy
|-- y_test.npy
`-- clients/
    |-- client_0_X.npy
    |-- client_0_y.npy
    |-- client_1_X.npy
    |-- client_1_y.npy
    |-- ...
    |-- client_9_X.npy
    `-- client_9_y.npy
```

The corresponding 20-client and 50-client directories use the same layout with 20 and 50 client pairs, respectively.

`metadata.json` must contain:

```json
{
  "classes": ["class_0", "class_1", "..."],
  "num_classes": 14,
  "num_features": 126,
  "num_clients": 10
}
```

The class names and their encoded ordering are read directly from `metadata.json`. For the 20-client and 50-client settings, change only the appropriate prepared partition and `num_clients` value.

The loader checks that `metadata.json` agrees with the requested `--num-clients` value.

### Dataset Source

The underlying dataset is **CCCS-CIC-AndMal-2020**, maintained by the Canadian Institute for Cybersecurity:

https://www.unb.ca/cic/datasets/andmal2020.html

Use and redistribution should follow the dataset provider's terms and the requirements of the research project or institution.

## Running FAM-STAR

Run commands from the repository root.

### 10 clients

```bash
python -m src.scripts.run_fam_star \
  --data-dir /path/to/cicandmal2020-10clients \
  --output-dir results/FAM-STAR/10_clients \
  --num-clients 10 \
  --num-rounds 50 \
  --num-epochs 5 \
  --batch-size 128 \
  --learning-rate 0.002 \
  --device cuda:0
```

### 20 clients

```bash
python -m src.scripts.run_fam_star \
  --data-dir /path/to/cicandmal2020-20clients \
  --output-dir results/FAM-STAR/20_clients \
  --num-clients 20 \
  --num-rounds 50 \
  --num-epochs 5 \
  --batch-size 128 \
  --learning-rate 0.002 \
  --device cuda:0
```

### 50 clients

```bash
python -m src.scripts.run_fam_star \
  --data-dir /path/to/cicandmal2020-50clients \
  --output-dir results/FAM-STAR/50_clients \
  --num-clients 50 \
  --num-rounds 50 \
  --num-epochs 5 \
  --batch-size 128 \
  --learning-rate 0.002 \
  --device cuda:0
```

The same Python source is used for all three federation sizes. Only the prepared data directory and `--num-clients` value change.

### Kaggle example

If the prepared data is attached to a Kaggle notebook, use its `/kaggle/input/...` path and write outputs to `/kaggle/working/...`:

```bash
python -m src.scripts.run_fam_star \
  --data-dir /kaggle/input/<prepared-fam-star-data> \
  --output-dir /kaggle/working/FAM_STAR_10clients_results \
  --num-clients 10 \
  --num-rounds 50 \
  --device cuda:0
```

## Configuration

The main configuration object is `FrameworkConfig` in `src/main.py`.

The command-line runner directly exposes:

- `--data-dir`
- `--output-dir`
- `--num-clients`
- `--num-rounds`
- `--num-epochs`
- `--batch-size`
- `--learning-rate`
- `--device`

FAM-STAR-specific hyperparameters such as sparsity, prune fraction, class-balancing coefficients, prototype coefficients, aggregation gamma, and quantization epsilon are defined in `FrameworkConfig` so that the experimental settings are explicit in one place.

## End-to-End Training Sequence

At communication round `t`, the implementation follows this order:

1. Activate the topology agreed at the end of the previous round.
2. Broadcast the current global model, current sparse mask, and available global class prototypes.
3. For every participating client:
   - load the current global state;
   - optimize the FAM-STAR local objective under the fixed current mask;
   - collect classification-gradient evidence;
   - construct a next-round sparse-topology proposal;
   - extract local class prototypes, support counts, and dispersion values;
   - quantize and serialize the local sparse update and prototype package;
   - retain the compression residual locally for error feedback.
4. Aggregate shared representation updates with diversity-adjusted client evidence.
5. Aggregate classifier rows with corresponding class-specific support.
6. Aggregate global class prototypes using support and dispersion.
7. Compute the sample-weighted Top-K consensus mask for the next round.
8. Evaluate the current-round global model on the shared held-out test set.
9. Save the round checkpoint and logged communication/predictive statistics.

The newly agreed topology is **pending state** and becomes active only at the start of the next communication round.

## Local Learning Objective

FAM-STAR uses four local components:

1. **CB-RS classification loss** for class imbalance and missing local classes.
2. **Masked FedProx** regularization against the current global model.
3. **Prototype alignment** between local embeddings and available global class prototypes.
4. **Prototype-head cross-entropy** using detached global prototypes.

Only the **CB-RS classification gradient** supplies topology evidence. Prototype, head, and proximal terms affect parameter learning but are not used for pruning/regrowth scores.

In round 1, no global prototypes are available yet, so prototype-based auxiliary terms are inactive. They become available after the first server prototype aggregation.

## Sparse Topology Adaptation

For each prunable shared tensor, FAM-STAR maintains a fixed active-connection budget.

- Active positions are ranked using the final local parameter magnitude multiplied by the final CB-RS gradient magnitude.
- A fraction `q = 0.05` of active positions with the lowest score is proposed for pruning.
- Inactive positions are ranked by the EMA of CB-RS gradient magnitude.
- The same number of positions is proposed for regrowth from the pre-pruning inactive pool.
- Newly pruned positions cannot be immediately regrown in the same proposal.
- The server combines client proposals using sample-weighted Top-K mask consensus.
- The classification head is kept dense.

This preserves the target 50% sparsity budget while allowing connectivity to evolve across rounds.

## Role-Specific Aggregation

FAM-STAR does not use one aggregation coefficient for all model parameters.

### Shared representation

Shared parameters blend:

- standard sample-count weighting; and
- diversity-adjusted class-support evidence.

The blend coefficient is `gamma = 0.50`.

### Classifier rows

Classifier row `c` blends:

- sample-count weighting; and
- support for class `c` across participating clients.

The resulting coefficients are normalized across clients before the class-specific row and bias are updated.

### Global class prototypes

For each available class, dequantized client prototypes are aggregated with the heuristic weight:

```text
log(1 + support) / (dispersion + epsilon)
```

If no client contributes a class in a round, the previous global prototype for that class is retained.

## Communication Accounting

FAM-STAR records **application-level serialized payload** rather than estimating communication only from active-parameter counts.

The uplink accounts for components such as:

- INT8 sparse update values;
- support bitmaps;
- quantization scales;
- next-round topology proposal bitmaps;
- quantized class prototypes;
- support/dispersion metadata;
- serialization metadata and overhead represented by the implementation.

The downlink accounts for the sparse FP32 global state, bit-packed current mask, and available global prototypes.

Communication is logged separately for uplink and downlink and can be summarized per round and per participating client.

## Output Structure

A run produces:

```text
results/FAM-STAR/10_clients/
|-- checkpoints/
|   |-- round_01_famstar.pt
|   |-- round_02_famstar.pt
|   |-- ...
|   `-- round_50_famstar.pt
|-- metrics/
|   |-- metrics_per_round.csv
|   |-- payload_events.csv
|   |-- communication_rounds.csv
|   |-- per_class_metrics.csv
|   `-- classification_report.txt
`-- figures/
    |-- metrics_per_round.png
    `-- confusion_matrix.png
```

The same output layout is used for 20 and 50 clients.

### Per-round checkpoints

Each checkpoint stores the round state required for research archival, including:

- global model parameters;
- current sparse masks;
- pending next-round masks, when applicable;
- client error-feedback residuals;
- global class prototypes;
- round metrics;
- Python, NumPy, PyTorch CPU, and available CUDA RNG states.

## Predictive Metrics

The global model is evaluated after every communication round on the shared held-out test set.

Logged metrics include:

- Accuracy.
- Balanced accuracy.
- Macro / micro / weighted precision.
- Macro / micro / weighted recall.
- Macro / micro / weighted F1.
- Worst-class F1.
- Training loss.
- Local and server execution time.
- Peak allocated GPU memory when CUDA is used.

The study uses **Macro-F1** as the primary predictive metric because it weights all 14 classes equally.

## Reported Round-50 Results

The following FAM-STAR values are reported at round 50:

| Clients | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-P | Weighted-R | Weighted-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.7303 | 0.6579 | 0.6806 | 0.6548 | 0.7592 | 0.7303 | 0.7359 |
| 20 | 0.7483 | 0.6436 | 0.6591 | 0.6380 | 0.7488 | 0.7483 | 0.7460 |
| 50 | 0.7362 | 0.6285 | 0.6125 | 0.6138 | 0.7273 | 0.7362 | 0.7294 |

Among the evaluated methods, FAM-STAR achieves the highest observed Macro-F1 and Weighted-F1 for all three federation sizes. Relative to the strongest baseline for Macro-F1, the reported improvements are **5.33**, **7.18**, and **2.36 percentage points** for 10, 20, and 50 clients, respectively.

## Reported Communication Results

Averaged over communication rounds 1-50 and equally across the 10-, 20-, and 50-client settings, the reported mean **per-client application-level payload per round** for FAM-STAR is:

| Direction | Mean payload (KB) |
|---|---:|
| Uplink | 598.806 |
| Downlink | 2685.374 |
| Total | 3284.180 |

Among the evaluated methods, FAM-STAR has the **lowest uplink payload** and the **second-lowest total payload**. The reported total-payload reductions are:

- **67.16%** versus FedAvg.
- **38.72%** versus FedDST.
- **43.21%** versus JoPEQ.
- **49.99%** versus Wireless Quantized FL.

Zero-shot pruning has a lower total payload in the reported comparison, while FAM-STAR achieves substantially higher Macro-F1 across all three federation sizes.

## Reproducibility Notes

- The reported study uses **one recorded seed (`42`)**.
- Client partitions are fixed for each federation size.
- All clients participate in every round.
- The shared test set is identical across methods for a given experimental study.
- The experiment is a logical federated simulation on one GPU, not a deployment across physical Android devices.
- For exact reproduction of the paper tables, use the archived prepared partitions and the reported configuration above.
- Small numerical differences may still occur across PyTorch/CUDA/cuDNN versions or different hardware environments.

## Scope and Limitations

The current evaluation assumes that the server and participating clients follow the federated protocol honestly.

The current scope does **not** claim robustness against:

- Byzantine clients;
- model or data poisoning;
- backdoor attacks;
- Sybil attacks;
- adversarial malware evasion;
- communication-channel attacks.

FAM-STAR also does not provide formal privacy guarantees such as differential privacy or secure aggregation, and it does not claim protection against information leakage from exchanged model updates or auxiliary class-level information.

The reported evidence is system-level and based on one recorded seed. Multi-seed evaluation, component-level ablations, and validation on additional Android datasets and deployment environments remain future work.

## Citation

If you use this implementation, please cite the accompanying manuscript:

> **FAM-STAR: Federated Android Malware Classification with Sparse Topology Adaptation and Role-Specific Aggregation**

Publication metadata and a formal BibTeX entry should be added here once the final venue metadata is available.

## License

See the repository `LICENSE` file for licensing terms.
