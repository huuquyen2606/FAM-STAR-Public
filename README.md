# FAM-STAR-Public

## FedAvg baseline: 10, 20, or 50 clients

Install the runtime dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

The local data-preparation command reads `data/raw/CICAndMal2020.csv` and writes the selected client partition under `data/processed/CICAndMal2020/<N>clients/`:

```powershell
python -m src.data.prepare_data --num-clients 10
python -m src.data.prepare_data --num-clients 20
python -m src.data.prepare_data --num-clients 50
```

Attach the matching public Kaggle dataset as an Input:

- 10 clients: [cicandmal2020-10clients](https://www.kaggle.com/datasets/zhacutis1tg/cicandmal2020-10clients)
- 20 clients: [cicandmal2020-20clients](https://www.kaggle.com/datasets/zhacutis1tg/cicandmal2020-20clients)
- 50 clients: [cicandmal2020-50clients](https://www.kaggle.com/datasets/zhacutis1tg/cicandmal2020-50clients)

In Kaggle, attach the selected dataset as an Input. The code maps `NUM_CLIENTS` to one fixed mount path and validates `metadata.json`. If the mount differs, edit the mapping in `src/baselines/common/data.py`; the FedAvg runner has no path flags.

Each dataset has `metadata.json`, `preprocessors.pkl`, `X_test.npy`, `y_test.npy`, and:

```text
clients/
  client_0_X.npy
  client_0_y.npy
  ...
  client_<N-1>_X.npy
  client_<N-1>_y.npy
```

Set `NUM_CLIENTS` in `src/baselines/fedavg.py` to 10, 20, or 50. `NUM_ROUNDS`, `NUM_EPOCHS`, `BATCH_SIZE`, `TEST_BATCH_SIZE`, `LEARNING_RATE`, and `SEED` are constants in the same file. The `--num-clients` switches above belong only to the separate data-preparation command.

Run FedAvg from the repository root:

```powershell
python -m src.baselines.fedavg
```

Output paths are selected automatically: `/kaggle/working/fedavg/<N>clients/` on Kaggle, or `experiments/fedavg/<N>clients/` locally. Change `resolve_output_dir()` in `src/baselines/fedavg.py` to change this behavior.

For a short smoke run (not a benchmark), temporarily set `NUM_ROUNDS = 1`, `NUM_EPOCHS = 1`, and `BATCH_SIZE = 4096` in `src/baselines/fedavg.py`, then run the same command. Restore the values before a full experiment.