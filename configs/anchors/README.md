# Anchor configs (prefix100 protocol)

This directory holds one anchor configuration per TSFM backbone
(`prefix100_<backbone>.json`), plus per-domain variants
(`prefix100_<backbone>_{liion,calb,naion,znion}.json`) that override the
backbone defaults for a single battery domain.

All anchors use the strict-protocol input population `observed_cycles: 100`
(the "prefix100" input regime) and were confirmed by staged grid searches over
the twelve-field space documented in `scripts/search/space.py`
(`learning_rate`, `weight_decay`, `dropout`, `sigma_soh`, `curve_loss`,
`compressed_dim`, `hidden_dim`, `shape_bins`, `field_dropout`,
`loss_weights[1]`, `loss_weights[2]`, `local_dim`).

Usage notes:

- An anchor is the starting point a TPE search perturbs (anchor + adjacent
  grid neighbors, see `scripts/search/run_tpe_asha.py --anchor-config`), or a
  directly trainable config for `train_eval_scripts/run_train.sh`.
- Omitted fields fall back to `batteryadapter.config.Config` defaults, which
  were checked field-by-field against the runs these files reproduce
  (`use_moe`, `use_interaction`, `use_soc`, `use_soh` default true;
  `bridge_read=attention`, `compressor=phase_cnn`, `life_loss=scaled_huber`,
  `task_head=auto`, `loss_weights=(1, 1, 0.1)`, `weight_decay=1e-4`).
- `seed` in each file is a template default (42); the official split seeds are
  2021/42/2024 for split domains and 2021 for LOAO domains, applied by the
  search drivers, not the config file.
