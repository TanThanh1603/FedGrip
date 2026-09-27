# FedGRIP

Code-only snapshot for federated domain generalization with pretrained backbones.
FedGRIP combines FedOMG-inspired gradient matching (GM), cross-domain quantile
transport (CQT), and head-calibrated pretrained preservation (HCPP).
HCPP combines head-only adaptation followed by fine-tuning with L2-SP.
This FedGRIP implementation does not use SWAD.

## Installation

Install a PyTorch/torchvision build appropriate for your GPU, then install the
core dependencies:

```bash
python -m pip install -r requirements.txt
```

The source environment used PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121.
The requirements file records core dependency versions from that environment;
installation in a fresh environment has not yet been verified. Third-party
reference projects may require their own additional dependencies.

## Contents

- `algorithm/client`, `algorithm/server`, `algorithm/aggregation`: algorithms.
- `model`, `utils`: model definitions and training utilities.
- `data/dataset.py`, `data/partition_data.py`: loading and partitioning.
- `shared_partitions.py`: shared partition support.
- `experiment_checkpoints.py`, `fdg_round_checkpoints.py`,
  `fdg_experiment_support.py`: experiment infrastructure.
- `tests`: native baseline and source-integrity checks.
- `third_party`: baseline reference sources and provenance manifests.

Datasets, pretrained weight files, experimental outputs, checkpoints, standalone
experiment runners, local result tables, IDE settings, and the original Git
history are intentionally excluded.

## Data

Obtain PACS, VLCS, Office-Home, and Terra Incognita from their original providers
under their respective terms. The general loader layout is
`data/<dataset>/raw/<domain>/<class>/<image>`; see the dataset and partition code
for dataset-specific handling. No datasets are distributed here.

## Entry points

`main.py` currently registers FedAvg, FedSR, GA, FedIIR, FedSAM, and StableFDG.
For example, inspect the arguments with:

```bash
python main.py --help
```

FedOMG and FedGRIP are provided as Python server/client implementations.
The original standalone experiment runners were deliberately excluded from
this snapshot. Consequently `main.py --algo FedGRIP` and `--algo FedOMG` are
not supported yet; this snapshot is not a complete one-command reproduction
package for those methods.

## Tests

```bash
python -m unittest discover -s tests -v
```

Runner-specific tests are excluded alongside the experiment runners.

## Attribution

This project builds on FedCCRL infrastructure and FedOMG aggregation, with
additional baseline integrations. See `NATIVE_FDG_BASELINES.md` and the upstream
READMEs, licenses, and `INSTALL_MANIFEST.json` files under `third_party/` for
source provenance. Third-party code retains its original terms; this snapshot
does not grant a new blanket license over upstream code.
