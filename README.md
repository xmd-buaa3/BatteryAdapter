# BatteryAdapter

Source files reside directly in this directory and in `backbones/`, `modules/`, `heads/`, `data/`, `training/`, and `evaluation/`. There is no enclosing `batteryadapter/` package directory.

The public scope is limited to model hyperparameters; native zero-shot forecasting, native LoRA, frozen-backbone BA training, and full backbone-plus-BA fine-tuning; information ablations; cross-domain transfer; and architecture ablations.

## Directory Structure

```text
.
├── config.py                   # Configuration fields and validation
├── model.py                    # BatteryAdapter model
├── checkpoint.py               # Checkpoint saving and loading
├── seeding.py                  # Random seeds
├── reference_inputs.py         # Reference input conversion
├── reference_baselines.py      # Cross-domain reference model interface
├── backbones/                  # Chronos-2, Moirai-2, MOMENT, Sundial, Toto-1
├── modules/                    # Input encoding and feature readout
├── heads/                      # Trajectory prediction heads
├── data/                       # Loading, preprocessing, and official splits
├── training/                   # Staged training, losses, and MMD
├── evaluation/                 # Evaluation
├── configs/
│   ├── models/                 # BA and native prediction settings for five models
│   ├── training/               # Model-by-domain configurations for four domains
│   ├── information/            # Information ablation controls
│   ├── transfer/               # Transfer protocol and parameter requirements
│   └── architecture/           # Architecture ablation controls
├── experimentals/              # Python entry points and shared implementations
├── train_eval_scripts/          # Seven corresponding sh entry points
├── deployment/                 # Dependencies and weight revisions for five backbones
└── docs/
    ├── model.md
    └── hyperparameters.md
```

`__init__.py` files are used only in subdirectories. Example imports are `from model import BatteryAdapter` and `from config import Config`. Run Python modules from this directory; the sh launchers locate this directory automatically.

## Environment and Data

Use Python from an existing environment, or install the dependencies listed in `requirements.txt`. Each real backbone uses the environment dependencies in `deployment/requirements/<model>.in`. The supported backbone identifiers are `chronos2`, `moirai2`, `moment`, `sundial`, and `toto1`.

Weight revisions and checksums are recorded in `deployment/manifests/model_revisions.json`. For BA entry points, `--model-root` is the shared weights directory, with each backbone subdirectory named after the final component of its Hugging Face repository ID. For native entry points, `--model-path` is the weights directory of the specific backbone.

## Python / sh Entry Points

| Experiment | Python module | sh script |
|---|---|---|
| Native zero-shot forecasting, without BA | `experimentals.run_zero_shot` | `run_zero_shot.sh` |
| Native LoRA, without BA | `experimentals.run_lora` | `run_lora.sh` |
| Frozen backbone, BA-only training | `experimentals.run_frozen_ba` | `run_frozen_ba.sh` |
| Full backbone-plus-BA fine-tuning | `experimentals.run_full_finetune` | `run_full_finetune.sh` |
| Information ablations | `experimentals.run_information_ablation` | `run_information_ablation.sh` |
| Cross-domain transfer | `experimentals.run_transfer` | `run_transfer.sh` |
| Architecture ablations | `experimentals.run_architecture_ablation` | `run_architecture_ablation.sh` |

See the [launcher documentation](train_eval_scripts/README.md) for parameters, invocation examples, and output formats. Model definitions are described in the [model documentation](docs/model.md), and parameter definitions in the [hyperparameter documentation](docs/hyperparameters.md).

## Training and Evaluation

BA entry points default to `train` and also support `search`, `evaluate`, `summarize`, and `matrix`. Training loads only train/val data and selects the Stage C checkpoint by validation MAPE. Testing uses a separate `evaluate` command and does not participate in hyperparameter selection. `search` runs exactly 12 TPE trials without ASHA; candidate ranges must be supplied explicitly in a `--space` JSON file, not inferred from test results.

The default observation limit is 100 cycles, with one observation window per battery. Batteries with fewer than 100 cycles retain their actual observation endpoints and masks. The model queries the absolute cycle grid 1–5000. SOH evaluation first restores the physical scale, pools valid points within each fold, and reports the mean and sample standard deviation across official folds within each domain. MAE is stored both in physical units and as a display value multiplied by 100; MAPE is reported as a percentage.

Use a new output path for each experiment; existing checkpoints and results are not overwritten. Frozen BA saves the required adapter parameters, while full fine-tuning saves the updated backbone parameters. Checkpoints do not provide exact training resumption.
