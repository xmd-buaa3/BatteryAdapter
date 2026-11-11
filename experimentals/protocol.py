"""Experiment matrix, input controls, and validation-based selection for the submission."""
from copy import deepcopy
from dataclasses import replace
import math
import statistics
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from config import Config
from data.dataset import BMFPrefixDataset
from data.protocol_extractors.bmf_strict import OFFICIAL_FOLDS
from data.protocol_extractors.common import source_eta
from training.trainer import Trainer, map_tensors

BACKBONES = ('chronos2', 'moirai2', 'moment', 'sundial', 'toto1')
ARCHITECTURE = ('no_caf', 'no_stage_a', 'no_atd', 'no_shape_experts')
INFORMATION = ('no_curves', 'no_aging', 'no_es', 'no_es_soh', 'no_es_anchor', 'no_es_soh_anchor')
INPUT_OVERRIDES = {
    'full': {},
    'no_curves': {'input_drop': ('curve',)},
    'no_aging': {'input_drop': ('aging',)},
    'no_es': {'input_drop': ('cyc',)},
    'no_es_soh': {'input_drop': ('cyc',), 'use_soh': False},
    'no_es_anchor': {'input_drop': ('cyc',), 'observed_start': False},
    'no_es_soh_anchor': {'input_drop': ('cyc',), 'use_soh': False, 'observed_start': False},
    'no_caf': {'compressor': 'capacity_only', 'use_soh': False,
               'input_drop': ('cyc', 'aging'), 'epochs': (0, 20, 30)},
    'no_stage_a': {'epochs': (0, 20, 30)},
    'no_atd': {'scheme_b': False},
    'no_shape_experts': {'use_moe': False},
}


def clone_inputs(inputs):
    return deepcopy(inputs)


def ba_config(base, *, model, seed, arm='full', full_finetune=False, stage_a=10):
    if model not in (*BACKBONES, 'tiny') or arm not in INPUT_OVERRIDES:
        raise ValueError('模型或消融不在论文清单内')
    if stage_a not in (5, 10):
        raise ValueError('论文 Stage A 只接受 5 或 10 epochs')
    if not isinstance(base, Config):
        base = Config(**base)
    values = dict(backbone=model, seed=seed, levels=(4,), input_protocol='batterymformer',
                  task_head='curve', observed_cycles=100, points=150, horizon=5000,
                  curve_channels=3, epochs=(stage_a, 20, 30), sigma_soh=.02,
                  state_bins=4, shape_experts=4, shape_bins=32, remaining_min=1.,
                  remaining_scale=900., start_margin=.001, shape_floor=.001,
                  backbone_train='full' if full_finetune else 'frozen',
                  backbone_init='pretrained', backbone_lora=False, use_tsfm=True,
                  use_moe=True, scheme_b=True, observed_start=True, use_soh=True,
                  information_ablation=None, input_drop=(), stage_c_compressor='joint',
                  stage_c_max_restarts=0, curve_loss='huber', cache_stage_b=False,
                  protect_life_head=False)
    values.update(INPUT_OVERRIDES[arm])
    return replace(base, **values)


def experiment_matrix():
    rows = []
    def add(experiment, model, fold=None, **parameters):
        parts = [experiment, model, fold or 'all'] + [str(value) for value in parameters.values()]
        rows.append(dict(id='/'.join(parts), experiment=experiment, model=model,
                         fold=fold, **parameters))
    for fold, (domain, seed) in OFFICIAL_FOLDS.items():
        for model in BACKBONES:
            for strategy in ('ZS', 'LoRA', 'BA', 'BA_FT'):
                add('adaptation', model, fold, domain=domain, seed=seed, strategy=strategy)
            for arm in ARCHITECTURE:
                add('architecture', model, fold, domain=domain, seed=seed, arm=arm)
        for model in ('toto1', 'sundial'):
            for arm in ('full', *INFORMATION):
                add('information', model, fold, domain=domain, seed=seed, arm=arm)
    source_folds = [fold for fold, (domain, _) in OFFICIAL_FOLDS.items() if domain == 'Li_ion']
    targets = [(fold, domain) for fold, (domain, _) in OFFICIAL_FOLDS.items() if domain != 'Li_ion']
    for source in source_folds:
        for fold, domain in targets:
            for model in ('BatteryMFormer', 'sundial', 'toto1'):
                scopes = ('model',) if model == 'BatteryMFormer' else ('ba', 'full')
                for strategy in ('Frozen', 'FT', 'DA'):
                    for scope in scopes:
                        if strategy == 'Frozen' and scope == 'full':
                            continue
                        add('transfer', model, fold, source_fold=source, domain=domain,
                            strategy=strategy, scope=scope)
    return rows


def validate_manifest(path):
    from data.protocol_extractors.bmf_strict import STRICT_CACHE_VERSION
    path = Path(path)
    spec = json.loads(path.read_text(encoding='utf-8'))
    tag = spec.get('split_tag')
    if (tag not in OFFICIAL_FOLDS or spec.get('protocol') != 'batterymformer'
            or spec.get('cache_version') != STRICT_CACHE_VERSION or spec.get('smoke_test')):
        raise ValueError('论文训练需要完整官方 fold 的严格缓存 manifest')
    seed = OFFICIAL_FOLDS[tag][1]
    if int(spec.get('split_seed', spec.get('seed', -1))) != seed:
        raise ValueError('manifest seed 与官方 fold 不符')
    asset = Path(__file__).resolve().parents[1] / 'data/protocol_splits' / f'batterymformer_{tag}.json'
    official = json.loads(asset.read_text(encoding='utf-8'))
    seen = set()
    for split in ('train', 'val', 'test'):
        expected = {(str(row['dataset']), str(row['cell_id'])) for row in official[split]}
        rows = spec.get(split, [])
        current = [(str(row['dataset']), str(row['cell_id'])) for row in rows]
        skipped = [(str(row.get('dataset', OFFICIAL_FOLDS[tag][0])), str(row.get('cell_id', Path(row['source_file']).stem)))
                   for row in spec.get('skipped', []) if row['split'] == split]
        members = current + skipped
        if not current or len(members) != len(set(members)) or set(members) != expected:
            raise ValueError(f'{split} 成员及明确排除项与官方 fold 不符')
        if seen.intersection(members):
            raise ValueError('官方 fold 的电池成员跨 split 重复')
        seen.update(members)
    return dict(fold=tag, domain=OFFICIAL_FOLDS[tag][0], seed=seed,
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())


class PaperPrefixDataset(BMFPrefixDataset):
    """Use one observation window per cell and preserve cached label validity."""
    def __init__(self, records, scaler, *, observed_cycles=100):
        super().__init__(records, scaler, 4, prefixes='observed', observed_cap=observed_cycles)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        record_index, prefix = self.index[index]
        original = self.records[record_index].targets
        mask = np.asarray(original.get('trajectory_mask', original['future_mask']), bool).copy()
        original_end = min(100, self.records[record_index].audit['raw_cycle_data_length'])
        if prefix < original_end:
            observed_valid = np.asarray(self.records[record_index].inputs['protocol_features']['soh_mask'], bool).reshape(-1)
            mask[prefix:original_end] = observed_valid[prefix:original_end]
        mask[:prefix] = False
        mask[self.records[record_index].audit['processed_soh_length']:] = False
        mask &= np.isfinite(np.asarray(original['future_soh']))
        sample['targets']['trajectory_mask'] = torch.as_tensor(mask)
        sample['targets']['future_mask'] = torch.as_tensor(mask.copy())
        return sample


def point_metrics(prediction, target, mask):
    prediction, target, mask = np.asarray(prediction, float), np.asarray(target, float), np.asarray(mask, bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('预测、标签和 mask 的形状必须相同')
    if not mask.any() or not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError('有效计分点为空或包含非有限值')
    if (target[mask] <= 0).any():
        raise ValueError('MAPE 需要正值物理 SOH 标签')
    error = np.abs(prediction[mask] - target[mask])
    absolute, relative, count = float(error.sum()), float((error / target[mask]).sum()), int(mask.sum())
    return dict(mae=absolute / count, mae_x1e2=100 * absolute / count,
                mape=100 * relative / count, abs_sum=absolute, relative_sum=relative, count=count)


@torch.no_grad()
def evaluate(model, loader):
    training = model.training
    model.eval()
    device = next(model.parameters()).device
    absolute = relative = 0.
    count = 0
    seen = set()
    try:
        for original in loader:
            batch = map_tensors(original, lambda tensor: tensor.to(device))
            keys = [(str(domain), str(cell), int(prefix)) for domain, cell, prefix
                    in zip(batch['domain'], batch['cell_id'], batch['prefix'])]
            if len(set(keys)) != len(keys) or seen.intersection(keys):
                raise ValueError('评估包含重复电池观测窗')
            seen.update(keys)
            query = model.query_grid(device)[None].expand_as(batch['targets']['query_cycles'])
            if not torch.equal(query, batch['targets']['query_cycles']):
                raise ValueError('论文评估需要绝对周期网格')
            prediction = model(batch['inputs'], 4)['future_soh'].double().cpu().numpy()
            target = batch['targets']['future_soh'].double().cpu().numpy()
            eta = np.asarray([source_eta(str(domain)) for domain in batch['domain']])[:, None]
            result = point_metrics(prediction * (1 - eta) + eta, target * (1 - eta) + eta,
                                   batch['targets']['trajectory_mask'].cpu().numpy())
            absolute += result['abs_sum']
            relative += result['relative_sum']
            count += result['count']
    finally:
        model.train(training)
    expected = {(str(loader.dataset.records[i].domain), str(loader.dataset.records[i].cell_id), int(k))
                for i, k in loader.dataset.index}
    if seen != expected or not count:
        raise ValueError('评估未覆盖全部登记的电池观测窗或无有效标签')
    return dict(mae=absolute / count, mae_x1e2=100 * absolute / count,
                mape=100 * relative / count, count=count, batteries=len(seen), aggregation='all-valid-points')


class PaperTrainer(Trainer):
    """Paper A/B/C budget with validation-MAPE selection and optional full-backbone updates."""
    def fit(self, train_loader, val_loader, *, scorer=None, patience=10, progress=None):
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError('patience 必须为正整数')
        if val_loader is None or not self.config.epochs[2]:
            raise ValueError('论文训练需要验证集和 Stage C')
        scorer = scorer or (lambda model, loader: evaluate(model, loader)['mape'])
        best = math.inf
        snapshot = None
        stale = 0
        self.checkpoint_selection = dict(checkpoint_policy='paper-best-c-val-mape',
                                         val_metric_unit='percent')
        for stage, epochs in zip('ABC', self.config.epochs):
            if not epochs:
                continue
            self.set_stage(stage)
            for epoch in range(1, epochs + 1):
                total = count = 0
                for batch in train_loader:
                    size = len(batch['cell_id'])
                    total += self.step(batch) * size
                    count += size
                if not count:
                    raise ValueError('论文训练集为空')
                row = dict(stage=stage, epoch=epoch, train_loss=total / count)
                if stage != 'A':
                    with torch.no_grad():
                        score = float(scorer(self.model, val_loader))
                    if not math.isfinite(score):
                        raise ValueError('验证 MAPE 非有限')
                    row['val_mape'] = score
                    if stage == 'C':
                        if score < best:
                            best, stale = score, 0
                            snapshot = self._snapshot()
                            self.checkpoint_selection.update(best_epoch=epoch, best_stage='C', best_val_mape=score)
                        else:
                            stale += 1
                self.history.append(row)
                if progress is not None:
                    progress(row)
                if stage == 'C' and stale >= patience:
                    self.checkpoint_selection['early_stopped'] = True
                    break
        if snapshot is None:
            raise ValueError('没有可供恢复的验证集赢家')
        self._restore(snapshot)
        return self.history


def fold_summary(rows, domain):
    expected = {fold for fold, (name, _) in OFFICIAL_FOLDS.items() if name == domain}
    folds = [row['fold'] for row in rows]
    if not expected or len(set(folds)) != len(folds) or set(folds) != expected:
        raise ValueError('域汇总需要全部官方 folds，不能缺失或重复')
    result = dict(domain=domain, folds=len(rows), std_ddof=1)
    for metric in ('mae', 'mape'):
        values = [float(row[metric]) for row in rows]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError('fold 指标必须为非负有限值')
        result[metric + '_mean'] = statistics.mean(values)
        result[metric + '_std'] = statistics.stdev(values)
    result['mae_mean_x1e2'] = 100 * result['mae_mean']
    result['mae_std_x1e2'] = 100 * result['mae_std']
    return result




def require_native_paper_support(model):
    if model not in BACKBONES:
        raise ValueError('模型不在论文清单内')
    raise ValueError('论文原生 ZS/LoRA 的相同输入协议尚未验证；不能以 SOH-only、重建线性外推或未训练预测头替代')
