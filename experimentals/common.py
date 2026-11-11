"""Paper experiment entry points: matrix generation does not start training, and search never reads test data."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from model import BatteryAdapter
from checkpoint import load
from data.dataset import Scaler, collate
from data.protocol_cache import load_protocol_cache
from seeding import set_deterministic_seed
from experimentals.protocol import (BACKBONES, INPUT_OVERRIDES, PaperPrefixDataset,
    PaperTrainer, ba_config, evaluate, experiment_matrix, fold_summary, validate_manifest)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def cache_root(manifest, supplied):
    if supplied is not None:
        return Path(supplied).resolve()
    spec = json.loads(Path(manifest).read_text(encoding='utf-8'))
    value = Path(spec.get('cache_root') or '.')
    return value if value.is_absolute() else Path(manifest).resolve().parent / value


def loaders_for(manifest, config, root, splits, batch_size, observed, scaler=None):
    records, protocol = load_protocol_cache(manifest, config, root, selected_splits=splits)
    if any(not rows for rows in records.values()):
        raise ValueError('论文实验的所需 split 不能为空')
    scaler = scaler or Scaler(metadata_profile=config.metadata_profile).fit(
        records['train'], split='train', observed_prefix=observed)
    loaders = {split: DataLoader(PaperPrefixDataset(rows, scaler, observed_cycles=observed),
                  batch_size=batch_size, shuffle=split == 'train', collate_fn=collate,
                  generator=torch.Generator().manual_seed(config.seed)) for split, rows in records.items()}
    return loaders, scaler, protocol


def train_one(manifest, config, output, *, root=None, batch_size=16, observed=100, device='cpu'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('训练输出目录已存在，不覆盖')
    set_deterministic_seed(config.seed)
    loaders, scaler, protocol = loaders_for(manifest, config, root, ('train', 'val'), batch_size, observed)
    model = BatteryAdapter(config, vocab_sizes={key: len(value) + 1 for key, value in scaler.vocab.items()})
    trainer = PaperTrainer(model, device)
    trainer.fit(loaders['train'], loaders['val'])
    result = dict(config=config.to_dict(), selection=trainer.checkpoint_selection,
                  validation=evaluate(model, loaders['val']), observation_cycles=observed,
                  batch_size=batch_size, evaluated_splits=['val'], formal_test_evaluated=False,
                  protocol=protocol, history=trainer.history)
    output.mkdir(parents=True)
    trainer.save(output / 'checkpoint.pt', scaler=scaler, protocol=protocol, run_metadata=result,
                 include_native=config.backbone_train == 'full')
    write_json(output / 'validation.json', result)
    return result


def search(args, config, identity):
    import optuna
    space = json.loads(args.space.read_text(encoding='utf-8'))
    allowed = {'learning_rate', 'weight_decay', 'dropout', 'batch_size', 'stage_a'}
    if (not isinstance(space, dict) or not space or set(space) - allowed
            or any(not isinstance(values, list) or not values for values in space.values())):
        raise ValueError('搜索空间需要非空候选列表，只允许 LR、weight_decay、dropout、batch_size、stage_a')
    for values in space.values():
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in values):
            raise ValueError('搜索候选需要有限数值')
    for key in ('batch_size', 'stage_a'):
        if key in space and any(not isinstance(value, int) or value < 1 for value in space[key]):
            raise ValueError('batch_size 与 stage_a 候选需要正整数')
    if 'stage_a' in space and any(value not in (5, 10) for value in space['stage_a']):
        raise ValueError('论文 Stage A 候选为 5 和 10')
    if args.output.exists():
        raise FileExistsError('搜索输出目录已存在，不覆盖')
    args.output.mkdir(parents=True)
    write_json(args.output / 'search.json', dict(identity=identity, trials=12, sampler='TPE',
               pruner='none', space=space, selection='validation MAPE', test_loaded=False))
    study = optuna.create_study(direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=identity['seed']), pruner=optuna.pruners.NopPruner())
    def objective(trial):
        sampled = {key: trial.suggest_categorical(key, values) for key, values in space.items()}
        batch = sampled.pop('batch_size', args.batch_size)
        stage_a = sampled.pop('stage_a', config.epochs[0])
        candidate = replace(config, **sampled,
                            epochs=(stage_a, 20, 30) if config.epochs[0] else (0, 20, 30))
        result = train_one(args.manifest, candidate, args.output / f'trial_{trial.number:02d}',
                           root=args.cache_root, batch_size=batch, observed=args.observed_cycles, device=args.device)
        return result['validation']['mape']
    study.optimize(objective, n_trials=12)
    trials = [dict(number=trial.number, value=trial.value, params=trial.params, state=trial.state.name)
              for trial in study.trials]
    best = study.best_trial
    write_json(args.output / 'winner.json', dict(identity=identity, best_trial=best.number,
        best_val_mape=best.value, checkpoint=str((args.output / f'trial_{best.number:02d}' / 'checkpoint.pt').resolve()),
        trials=trials, successful_trials=12, test_loaded=False, requires_separate_test=True))


def evaluate_checkpoint(args, identity, loaded=None):
    model, scaler, payload = loaded if loaded is not None else load(args.checkpoint, device=args.device)
    model.to(args.device)
    if model.config.backbone not in BACKBONES:
        raise ValueError('checkpoint 骨干不在论文清单内')
    if payload['run'].get('selection', {}).get('checkpoint_policy') != 'paper-best-c-val-mape':
        raise ValueError('论文评估需要按论文协议验证选优的 checkpoint，不能混用历史预算')
    if payload['protocol'].get('manifest_sha256') != identity['manifest_sha256']:
        raise ValueError('checkpoint 与当前 fold manifest 不匹配；跨域测试使用迁移入口')
    loaders, _, _ = loaders_for(args.manifest, model.config, args.cache_root, ('test',),
                               args.batch_size, args.observed_cycles, scaler)
    result = dict(identity=identity, config=model.config.to_dict(), metrics=evaluate(model, loaders['test']),
                  checkpoint=str(args.checkpoint), observation_cycles=args.observed_cycles,
                  test_selection=False)
    write_json(args.output, result)






def main(argv=None, *, full_finetune=False, models=BACKBONES, arms=('full',), variants=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='train', choices=('matrix', 'train', 'search', 'evaluate', 'summarize'))
    parser.add_argument('--model', choices=models)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--space', type=Path)
    parser.add_argument('--results', nargs='+', type=Path)
    parser.add_argument('--domain')
    parser.add_argument('--arm', choices=arms, default='full')
    parser.add_argument('--stage-a', type=int, choices=(5, 10), default=10)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.observed_cycles = 100
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('batch-size 必须为正整数')
    if args.action == 'matrix':
        write_json(args.output, dict(jobs=experiment_matrix(), matrix_only=True, results_reproduced=False,
            blocked=['原生 ZS/LoRA 相同输入协议', '去 CAF 的周期容量输入', 'BatteryMFormer 跨域迁移']))
        return
    if args.action == 'summarize':
        if not args.results or not args.domain:
            parser.error('summarize 需要 --results 和 --domain')
        data = [json.loads(path.read_text(encoding='utf-8')) for path in args.results]
        groups = {(row['config']['backbone'], row['observation_cycles'], row['config']['backbone_train'],
                   tuple(row['config']['input_drop']), row['config']['compressor'], row['config']['use_soh'],
                   row['config']['observed_start'], row['config']['scheme_b'], row['config']['use_moe'],
                   tuple(row['config']['epochs'])) for row in data}
        if len(groups) != 1:
            raise ValueError('汇总不能混合不同模型、输入组合或预算')
        rows = [dict(fold=row['identity']['fold'], **row['metrics']) for row in data]
        write_json(args.output, fold_summary(rows, args.domain))
        return
    if args.manifest is None:
        parser.error('该实验需要 --manifest')
    identity = validate_manifest(args.manifest)
    args.cache_root = cache_root(args.manifest, args.cache_root)
    if args.model_root is not None:
        os.environ['BATTERYADAPTER_MODEL_ROOT'] = str(args.model_root.resolve())
    if args.action == 'evaluate':
        if args.checkpoint is None:
            parser.error('evaluate 需要 --checkpoint')
        args.batch_size = args.batch_size or 16
        loaded = load(args.checkpoint, device='cpu')
        model = loaded[0]
        if model.config.backbone not in models or model.config.backbone_train != ('full' if full_finetune else 'frozen'):
            raise ValueError('checkpoint 与入口的模型或骨干更新范围不同')
        candidate = ba_config(model.config, model=model.config.backbone, seed=identity['seed'],
                              arm=args.arm, full_finetune=full_finetune, stage_a=model.config.epochs[0] or 10)
        if candidate.to_dict() != model.config.to_dict():
            raise ValueError('checkpoint 与所选消融或论文配置不同')
        evaluate_checkpoint(args, identity, loaded)
        return
    if args.model is None:
        parser.error('train/search 需要 --model')
    domain_name = {'Li_ion': 'liion', 'CALB': 'calb', 'NA-ion': 'naion', 'ZN-coin': 'znion'}[identity['domain']]
    path = args.config or ROOT / 'configs/training' / args.model / f'{domain_name}.json'
    settings = json.loads(path.read_text(encoding='utf-8'))
    base = settings.get('config', settings.get('ba', settings))
    args.batch_size = args.batch_size or settings.get('batch_size', 16)
    if variants is not None:
        declared = json.loads((ROOT / variants).read_text(encoding='utf-8'))[args.arm] if args.arm != 'full' else {}
        expected = json.loads(json.dumps(INPUT_OVERRIDES[args.arm]))
        if declared != expected:
            raise ValueError('消融配置文件与论文开关定义不同')
    config = ba_config(base, model=args.model, seed=identity['seed'], arm=args.arm,
                       full_finetune=full_finetune, stage_a=args.stage_a)
    if args.action == 'search':
        if args.space is None:
            parser.error('search 需要 --space，不能推断论文未提供的搜索范围')
        search(args, config, identity)
    else:
        train_one(args.manifest, config, args.output, root=args.cache_root,
                  batch_size=args.batch_size, observed=100, device=args.device)
        write_json(args.output / 'identity.json', identity)


if __name__ == '__main__':
    main()
