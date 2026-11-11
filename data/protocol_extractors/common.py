"""Built-in raw-data protocol extraction matching the reference loaders."""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np

OBSERVED_CYCLES, CURVE_POINTS = 100, 300

class SkipCell(ValueError):
    pass

def discover_cells(dataset_root, dataset=None):
    root = Path(dataset_root)
    names = [dataset] if dataset else [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    return [(n, p) for n in names for p in sorted((root / n).glob('*.pkl'))]

def source_eta(dataset):
    return .9 if str(dataset).upper() == 'CALB' else .8

def _finite(x):
    x = np.asarray(x, float).reshape(-1)
    return np.nan_to_num(x, nan=0., posinf=0., neginf=0.)

def _index(x, n=150):
    x = _finite(x)
    if not x.size: return np.zeros(n, np.float32)
    return np.interp(np.linspace(1, len(x)+1, n), np.arange(1, len(x)+1), x).astype(np.float32)

def _soc(v, i, c, s, n, charge):
    v, i, c, s = [_finite(x) for x in (v, i, c, s)]
    if not s.size or not v.size: return (np.zeros(n, np.float32),) * 4
    sm = np.maximum.accumulate(s) if charge else np.minimum.accumulate(s)
    d = np.diff(sm); good = np.r_[True, d > 1e-8] if charge else np.r_[True, d < -1e-8]
    if good.sum() < 2: return tuple(_index(x, n) for x in (v, i, c, s))
    ss, vv, ii, cc = sm[good], v[good], i[good], c[good]
    t = np.linspace(ss.min(), ss.max(), n) if charge else np.linspace(ss.max(), ss.min(), n)
    if not charge: t0, ss, vv, ii, cc = t[::-1], ss[::-1], vv[::-1], ii[::-1], cc[::-1]
    else: t0 = t
    return (np.interp(t0, ss, vv).astype(np.float32), np.interp(t0, ss, ii).astype(np.float32),
            np.interp(t0, ss, cc).astype(np.float32), t.astype(np.float32))

def _cycle(item, nominal, file_name, dataset, bmf, interval):
    v, i = _finite(item['voltage_in_V']), _finite(item['current_in_A'])
    cq, dq, tm = _finite(item['charge_capacity_in_Ah']), _finite(item['discharge_capacity_in_Ah']), _finite(item['time_in_s']) / 3600.
    if not (len(v) == len(i) == len(cq) == len(dq) == len(tm)): raise SkipCell('unaligned cycle arrays')
    rate = i / nominal; ci, di = np.flatnonzero(rate >= .01), np.flatnonzero(rate <= -.01)
    if not ci.size or not di.size: raise SkipCell('missing charge/discharge .01C segment')
    ce, de = ci[-1], di[-1]; prefix = file_name.split('_')[0]
    if prefix == 'CALB': prefix = '_'.join(file_name.split('_')[:2])
    first_discharge = prefix in {'RWTH','CALB_0','CALB_25','CALB_45'} or prefix == 'ZN-coin'
    if first_discharge:
        vd, cd, id_, td = v[:de], dq[:de], i[:de], tm[:de]
        vc, cc, ic, tc = v[de:], cq[de:], i[de:], tm[de:]; keep = np.abs(ic/nominal) > .01
        vc, cc, ic, tc = vc[keep], cc[keep], ic[keep], tc[keep]
    else:
        vd, cd, id_, td = v[ce:], dq[ce:], i[ce:], tm[ce:]; keep = np.abs(id_/nominal) > .01
        vd, cd, id_, td = vd[keep], cd[keep], id_[keep], td[keep]
        vc, cc, ic, tc = v[:ce], cq[:ce], i[:ce], tm[:ce]
    if min(len(vc), len(vd), len(cc), len(cd)) < 2: raise SkipCell('short phase')
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    cee = (float(np.max(cd)/max(np.max(cc),1e-6)), float(integrate(vd*np.abs(id_),td)/max(integrate(vc*np.abs(ic),tc),1e-6)))
    lo, hi = interval; span = hi-lo
    sc = lo + (cc-cc[0])/max(float(np.ptp(cc)),1e-6)*span; sd = hi - (cd-cd[0])/max(float(np.ptp(cd)),1e-6)*span
    if bmf: vc,ic,cc,sc = _soc(vc,ic,cc,sc,150,True); vd,id_,cd,sd = _soc(vd,id_,cd,sd,150,False)
    else: vc,ic,cc,sc = _index(vc),_index(ic),_index(cc),_index(sc); vd,id_,cd,sd = _index(vd),_index(id_),_index(cd),_index(sd)
    voltage, current, capacity = np.r_[vc,vd], np.r_[ic,id_]/nominal, np.r_[cc,cd]
    if not bmf: voltage /= max(float(np.max(voltage)),1e-6); capacity /= nominal
    return np.stack((voltage,current,capacity)).astype(np.float32), np.r_[sc,sd].astype(np.float32), cee

def _raw(path, dataset, bmf):
    with Path(path).open('rb') as f: raw = pickle.load(f)
    nominal = float(raw.get('nominal_capacity_in_Ah', np.nan)); interval = np.asarray(raw.get('SOC_interval',(0.,1.)),float).reshape(-1)
    records = raw.get('cycle_data')
    if not np.isfinite(nominal) or nominal <= 0 or interval.size < 2 or interval[1] <= interval[0]: raise SkipCell('invalid nominal/SOC interval')
    if not isinstance(records,list) or not records: raise SkipCell('empty cycle_data')
    curves=np.zeros((100,3,300),np.float32); soc=np.zeros((100,300),np.float32); cee=np.zeros((100,2),np.float32)
    curve_mask=np.zeros_like(curves,bool); cycle_mask=np.zeros(100,bool); cee_mask=np.zeros_like(cee,bool)
    parsed=0
    # Both reference loaders use list position; CALB records may start at physical cycle 2.
    for ix,item in enumerate(records[:OBSERVED_CYCLES]):
        slot=ix
        try:
            curves[slot],soc[slot],cee[slot]=_cycle(item,nominal,Path(path).name,dataset,bmf,interval[:2])
        except (AttributeError, KeyError, SkipCell, TypeError, ValueError):
            continue
        parsed += 1; cycle_mask[slot]=True; curve_mask[slot]=True; cee_mask[slot]=True
    if not parsed: raise SkipCell('no valid cycle curves')
    observed_end=min(len(records), OBSERVED_CYCLES)
    return raw,curves,soc,cee,curve_mask,cycle_mask,cee_mask,parsed,observed_end

def _soh(root,dataset,path):
    folder='total_MICH' if dataset=='MICH_EXP' else dataset; p=Path(root)/folder/Path(path).name
    if not p.is_file() and Path(path).name.startswith('MICH_'):
        p = Path(root) / 'total_MICH' / Path(path).name
    if not p.is_file() and Path(path).name.startswith('Stanford'):
        p = Path(root) / 'Stanford_2' / Path(path).name
    if not p.is_file():
        matches = sorted(Path(root).rglob(Path(path).name))
        if len(matches) == 1:
            p = matches[0]
    if not p.is_file(): raise SkipCell(f'processed SOH missing: {p}')
    with p.open('rb') as f: x=pickle.load(f)
    y=np.asarray(x.get('SOH'),np.float32).reshape(-1)
    if not y.size or not np.isfinite(y).all(): raise SkipCell('invalid processed SOH')
    return y,str(p)

def _embedding(path,name):
    with Path(path).open('rb') as f: m=pickle.load(f)
    x=np.asarray(m.get(name),np.float32).reshape(-1)
    if x.shape != (1024,) or not np.isfinite(x).all(): raise SkipCell('missing prompt embedding')
    return x.reshape(1,1024),str(path)

def extract_batterymformer(path, *, dataset, processed_soh_root, prompt_embeddings_path):
    path=Path(path); soh,soh_src=_soh(processed_soh_root,dataset,path)
    if len(soh)<=100: raise SkipCell('EOL <= 100')
    raw,curves,soc,cee,curve_mask,cycle_mask,cee_mask,parsed,observed_end=_raw(path,dataset,True); emb,emb_src=_embedding(prompt_embeddings_path,path.name); eta=source_eta(dataset)
    y=(soh-eta)/(1-eta); future=np.zeros(5000,np.float32); mask=np.zeros(5000,bool); n=min(5000,len(y)); future[:n]=y[:n]; mask[observed_end:n]=True
    soh_mask=(np.arange(100)[:,None] < observed_end) & (np.arange(100)[:,None] < len(y))
    start_slot=max(0, observed_end-1); start_valid=start_slot < len(y)
    provenance={'source_eta':eta,'soh_source':soh_src,'embedding_source':emb_src,'raw_cycle_data_length':len(raw['cycle_data']),'parsed_curve_count':parsed,'processed_soh_length':len(soh),'observed_cycle_end':observed_end}
    return dict(curves=curves,curve_mask=curve_mask,cycle_mask=cycle_mask,capacity=np.zeros((100,1),np.float32),capacity_mask=np.zeros((100,1),bool),soc_input=soc,soc_mask=curve_mask.any(axis=1),soh_input=y[:100,None] if len(y)>=100 else np.pad(y,(0,100-len(y)))[:,None],soh_mask=soh_mask,cycle_level_features=cee,cycle_level_features_mask=cee_mask,aging_condition_embedding=emb,aging_condition_mask=np.ones((1,1024),bool),protocol_feature_mask=np.asarray(True),future_soh=future,future_mask=mask,trajectory_mask=mask,total_life=np.asarray(float(len(soh)),np.float32),start_soh=np.asarray(y[start_slot] if start_valid else np.nan,np.float32),start_soh_mask=np.asarray(start_valid,bool),raw_cycle_data_length=np.asarray(len(raw['cycle_data']),np.int64),parsed_curve_count=np.asarray(parsed,np.int64),processed_soh_length=np.asarray(len(soh),np.int64),normalization_version=np.asarray(f'batterymformer-soh-{eta:g}-v1'),raw_source=np.asarray(str(path)),provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)))

def _label(path,root):
    name=path.name
    prefix = ('total_MICH' if name.startswith('MICH') else
              'Tongji' if name.startswith('Tongji') else name.split('_')[0])
    p=Path(root)/f'{prefix}_labels.json'
    if not p.is_file(): raise SkipCell(f'life labels missing: {p}')
    labels=json.loads(p.read_text(encoding='utf-8')); key=name.replace('--','-#') if name.startswith('Tongji') else name
    if key not in labels or not np.isfinite(labels[key]): raise SkipCell('missing real EOL')
    return float(labels[key]),str(p)

def extract_batterylife(path, *, dataset, life_labels_root, processed_soh_root=None):
    path=Path(path); life,src=_label(path,life_labels_root)
    if dataset.upper()=='CALB' and processed_soh_root is not None:
        try: life=float(len(_soh(processed_soh_root,dataset,path)[0]))
        except SkipCell: pass
    if life<=100: raise SkipCell('EOL <= 100')
    raw,curves,_,_,curve_mask,cycle_mask,_,parsed,observed_end=_raw(path,dataset,False); eta=source_eta(dataset)
    processed_soh_len=-1
    if processed_soh_root is not None:
        try: processed_soh_len=len(_soh(processed_soh_root,dataset,path)[0])
        except SkipCell: pass
    provenance={'source_eta':eta,'label_source':src,'raw_cycle_data_length':len(raw['cycle_data']),'parsed_curve_count':parsed,'processed_soh_length':processed_soh_len,'observed_cycle_end':observed_end}
    return dict(curves=curves,curve_mask=curve_mask,cycle_mask=cycle_mask,capacity=np.zeros((100,1),np.float32),capacity_mask=np.zeros((100,1),bool),total_life=np.asarray(life,np.float32),future_soh=np.zeros(5000,np.float32),future_mask=np.zeros(5000,bool),trajectory_mask=np.zeros(5000,bool),start_soh=np.asarray(np.nan,np.float32),start_soh_mask=np.asarray(False),protocol_feature_mask=np.asarray(False),raw_cycle_data_length=np.asarray(len(raw['cycle_data']),np.int64),parsed_curve_count=np.asarray(parsed,np.int64),processed_soh_length=np.asarray(processed_soh_len,np.int64),normalization_version=np.asarray('batterylife-v1'),raw_source=np.asarray(str(path)),provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)))

def extract_cell(*args,**kwargs):
    raise TypeError('use extract_batterymformer or extract_batterylife')
