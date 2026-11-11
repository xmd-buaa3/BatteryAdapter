"""Frozen, FT, and DA transfer from Li-ion to CALB, Na-ion, and Zn-ion.

Training and MMD sampling follow flow_final/batteryadapter/training/transfer.py.
The public entry point uses paper observation windows and validation-based selection;
it does not read historical server job registrations.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader, RandomSampler

from checkpoint import load, save
from data.dataset import collate
from seeding import set_deterministic_seed
from training.domain_adaptation import MMDLoss
from training.losses import losses
from training.trainer import map_tensors
from experimentals.protocol import OFFICIAL_FOLDS, evaluate, validate_manifest
from experimentals.common import cache_root, loaders_for, write_json


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def adapt_model(model, source_loader, target_loader, val_loader, *, strategy, epochs,
                patience, learning_rate, backbone_lr, da_weight):
    native_ids = {id(parameter) for parameter in model.backbone.parameters()}
    native = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in native_ids]
    adapter = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in native_ids]
    groups = [{'params': adapter, 'lr': learning_rate}]
    if native:
        groups.append({'params': native, 'lr': backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=model.config.weight_decay)
    device = next(model.parameters()).device
    mmd = MMDLoss()
    best = math.inf
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        pairs = zip(source_loader, target_loader) if strategy == 'DA' else ((None, batch) for batch in target_loader)
        total_sum = count = 0
        for source_batch, target_batch in pairs:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            target_batch = map_tensors(target_batch, lambda tensor: tensor.to(device))
            predicted = model(target_batch['inputs'], 4, return_embedding=strategy == 'DA')
            total = losses(predicted, target_batch['targets'], model.config)['total']
            if strategy == 'DA':
                source_batch = map_tensors(source_batch, lambda tensor: tensor.to(device))
                source_prediction = model(source_batch['inputs'], 4, return_embedding=True)
                total = (total + losses(source_prediction, source_batch['targets'], model.config)['total']
                         + da_weight * mmd(source_prediction['embedding'], predicted['embedding']))
            if not torch.isfinite(total):
                raise FloatingPointError('迁移损失非有限')
            total.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad],
                                          5., error_if_nonfinite=True)
            optimizer.step()
            total_sum += float(total.detach()); count += 1
        if not count:
            raise ValueError('迁移训练样本为空')
        score = evaluate(model, val_loader)['mape']
        history.append(dict(epoch=epoch, loss=total_sum/count, val_mape=score))
        if score < best:
            best, stale = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError('迁移训练未得到验证集赢家')
    model.load_state_dict(best_state, strict=True)
    return history, dict(best_val_mape=best, best_epoch=best_epoch, patience=patience, epochs_limit=epochs)


def summarize(paths, domain, output):
    rows = [json.loads(Path(path).read_text(encoding='utf-8')) for path in paths]
    groups = {(row['model'], row['strategy'], row['scope']) for row in rows}
    if len(groups) != 1:
        raise ValueError('迁移汇总不能混合不同模型、策略或骨干更新范围')
    sources = {fold for fold, (name, _) in OFFICIAL_FOLDS.items() if name == 'Li_ion'}
    targets = {fold for fold, (name, _) in OFFICIAL_FOLDS.items() if name == domain}
    keys = [(row['source']['fold'], row['target']['fold']) for row in rows]
    if not targets or domain == 'Li_ion' or len(set(keys)) != len(keys) or set(keys) != {(s, t) for s in sources for t in targets}:
        raise ValueError('迁移汇总需要三个源模型乘全部目标 folds，不能缺失或重复')
    result = dict(model=rows[0]['model'], strategy=rows[0]['strategy'], scope=rows[0]['scope'],
                  domain=domain, source_models=3, target_folds=len(targets), std_ddof=1)
    for metric in ('mae', 'mape'):
        values = [statistics.mean(row['metrics'][metric] for row in rows if row['source']['fold'] == source)
                  for source in sorted(sources)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('迁移指标非有限')
        result[metric + '_mean'] = statistics.mean(values)
        result[metric + '_std'] = statistics.stdev(values)
    result['mae_mean_x1e2'] = 100 * result['mae_mean']
    result['mae_std_x1e2'] = 100 * result['mae_std']
    write_json(output, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--target-manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--strategy', choices=('Frozen', 'FT', 'DA'))
    parser.add_argument('--scope', choices=('ba', 'full'), default='ba')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--backbone-lr', type=float)
    parser.add_argument('--da-weight', type=float)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-root', type=Path)
    parser.add_argument('--results', nargs='+', type=Path)
    parser.add_argument('--domain')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.results:
        if not args.domain:
            parser.error('迁移汇总需要 --domain')
        summarize(args.results, args.domain, args.output)
        return
    if not all((args.source_manifest, args.target_manifest, args.checkpoint, args.strategy)):
        parser.error('迁移需要 source-manifest、target-manifest、checkpoint、strategy')
    if args.strategy == 'Frozen' and args.scope == 'full':
        parser.error('论文 BA_FT 没有 Frozen 迁移结果')
    if args.strategy != 'Frozen':
        if any(value is None or value < 1 for value in (args.epochs, args.patience, args.batch_size)):
            parser.error('FT/DA 需要明确的正整数 epochs、patience、batch-size')
        if any(value is None or not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.backbone_lr)):
            parser.error('FT/DA 需要明确的有限正值 learning-rate 与 backbone-lr')
        if args.strategy == 'DA' and (args.da_weight is None or not math.isfinite(args.da_weight) or args.da_weight <= 0):
            parser.error('DA 需要明确的正值 MMD 权重')
    if args.batch_size < 1 or args.output.exists():
        raise ValueError('batch-size 必须为正，输出路径不能已存在')
    source = validate_manifest(args.source_manifest)
    target = validate_manifest(args.target_manifest)
    if source['domain'] != 'Li_ion' or target['domain'] == 'Li_ion':
        raise ValueError('论文迁移只从 Li-ion 到另外三个域')
    if args.model_root:
        os.environ['BATTERYADAPTER_MODEL_ROOT'] = str(args.model_root.resolve())
    checkpoint_hash = digest(args.checkpoint)
    model, source_scaler, payload = load(args.checkpoint, device=args.device)
    if model.config.backbone not in ('sundial', 'toto1'):
        raise ValueError('该入口只覆盖论文中的 Sundial/Toto-1；不以 BA 替代 BatteryMFormer')
    if (payload['protocol'].get('manifest_sha256') != source['manifest_sha256']
            or model.config.seed != source['seed']
            or payload['run'].get('selection', {}).get('checkpoint_policy') != 'paper-best-c-val-mape'):
        raise ValueError('源 checkpoint 不是该 Li-ion fold 的论文验证赢家')
    expected_scope = 'full' if model.config.backbone_train == 'full' else 'ba'
    if args.scope != expected_scope:
        raise ValueError('源模型的 BA/BA_FT 类型与指定 scope 不符')
    set_deterministic_seed(source['seed'])
    history = []
    selection = None
    scaler = source_scaler
    if args.strategy != 'Frozen':
        configured = replace(model.config, learning_rate=args.learning_rate, backbone_lr=args.backbone_lr)
        original = model.config
        for module in model.modules():
            if getattr(module, 'config', None) is original:
                module.config = configured
        model.config = configured
        model.requires_grad_(True)
        model.backbone.trainable = args.scope == 'full'
        model.backbone.requires_grad_(args.scope == 'full')
        target_loaders, scaler, target_protocol = loaders_for(args.target_manifest, configured,
            cache_root(args.target_manifest, None), ('train', 'val'), args.batch_size, 100,
            source_scaler if args.strategy == 'DA' else None)
        source_loader = None
        target_loader = target_loaders['train']
        if args.strategy == 'DA':
            source_loaders, _, _ = loaders_for(args.source_manifest, configured,
                cache_root(args.source_manifest, None), ('train',), args.batch_size, 100, source_scaler)
            source_loader = source_loaders['train']
            sampler = RandomSampler(target_loader.dataset, replacement=True, num_samples=len(source_loader.dataset),
                                    generator=torch.Generator().manual_seed(source['seed'] + 1))
            target_loader = DataLoader(target_loader.dataset, sampler=sampler,
                                       batch_size=args.batch_size, collate_fn=collate)
        history, selection = adapt_model(model, source_loader, target_loader, target_loaders['val'],
            strategy=args.strategy, epochs=args.epochs, patience=args.patience,
            learning_rate=args.learning_rate, backbone_lr=args.backbone_lr, da_weight=args.da_weight or 0.)
    tests, _, target_protocol = loaders_for(args.target_manifest, model.config,
        cache_root(args.target_manifest, None), ('test',), args.batch_size, 100, scaler)
    result = dict(source=source, target=target, model=model.config.backbone, strategy=args.strategy,
        scope=args.scope, source_checkpoint_sha256=checkpoint_hash, selection=selection,
        metrics=evaluate(model, tests['test']), history=history, target_labels_used=args.strategy != 'Frozen',
        source_labels_used=args.strategy == 'DA', da_weight=args.da_weight)
    if digest(args.checkpoint) != checkpoint_hash:
        raise ValueError('源 checkpoint 在迁移过程中发生变化')
    args.output.mkdir(parents=True)
    if args.strategy != 'Frozen':
        save(args.output / 'checkpoint.pt', model, scaler=scaler, protocol=target_protocol,
             history=history, stage='adaptation', run_metadata={'transfer': result}, include_native=args.scope == 'full')
    write_json(args.output / 'result.json', result)


if __name__ == '__main__':
    main()
