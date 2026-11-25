"""Build indices for verified MR-to-US exports. All datalist paths are relative."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

import numpy as np

SPLITS = {'train': 'training', 'val': 'validation', 'test': 'test'}


def validate_splits(cases):
    patients = {}
    for case in cases:
        split, patient = case['split'], case['patient']
        if split not in SPLITS:
            raise ValueError('MONAI export requires train, val or test in each split field')
        if patients.setdefault(patient, split) != split:
            raise ValueError(f'Patient appears in multiple splits: {patient}')


def save_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def patient_means(rows, key):
    values = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            values[row['patient']].append(row[key])
    return [mean(group) for group in values.values()]


def metric_rows(split, pairs, targets):
    def distribution(metric, stage, weighting, unit, values):
        stats = dict(n=len(values), mean=None, sd=None, median=None, p90=None, p95=None, min=None, max=None)
        if values:
            stats.update(mean=mean(values), sd=float(np.std(values, ddof=1)) if len(values)>1 else None,
                         median=float(np.median(values)), p90=float(np.percentile(values,90)),
                         p95=float(np.percentile(values,95)), min=min(values), max=max(values))
        return dict(split=split, metric=metric, stage=stage, unit=unit, weighting=weighting, **stats)

    for stage in ('before','after'):
        key = f'target_tre_{stage}_mm'
        yield distribution('target_TRE', stage, 'patient_equal', 'mm', patient_means(pairs, key))
        yield distribution('target_TRE', stage, 'pair_equal', 'mm', [p[key] for p in pairs if p[key] is not None])
        yield distribution('target_TRE', stage, 'individual_target', 'mm', [t[f'{stage}_mm'] for t in targets])
        for metric, column, unit in [('gland_Dice', f'dice_{stage}', 'fraction'),
                                     ('gland_centroid_distance', f'gland_centroid_{stage}_mm', 'mm'),
                                     ('target_plus_gland_centroid_distance', f'mixed_centroid_{stage}_mm', 'mm')]:
            yield distribution(metric, stage, 'pair_equal', unit, [p[column] for p in pairs])
    for metric, key in [('gland_ASSD', 'symmetric_surface_distance_output_grid_mm'), ('gland_HD95','hd95_output_grid_mm')]:
        yield distribution(metric, 'after', 'pair_equal', 'mm', [p[key] for p in pairs if p[key] is not None])


def write_export(root, cases, results, config):
    """Write a complete MONAI datalist, split-level checks and metric tables."""
    index = {c['pair_id']: c for c in cases}
    if len(results) != len(index) or {r['pair_id'] for r in results} != index.keys():
        raise ValueError('Cannot index an incomplete batch')
    groups = {name: [] for name in SPLITS.values()}
    pairs, targets = [], []
    for row in sorted(results, key=lambda r: r['pair_id']):
        case = index[row['pair_id']]
        folder = Path(case['relative_output'])
        report = json.loads((root / folder / 'registration.json').read_text())
        if report['verification']['status'] != 'passed' or report['case_id'] != case['pair_id']:
            raise ValueError(f'Unverified export: {folder}')
        if report['config'] != json.loads(json.dumps(config.to_dict())):
            raise ValueError(f'Configuration mismatch: {folder}')
        record = dict(pair_id=case['pair_id'], patient=case['patient'], source_qc=case['source_qc'],
                      requires_review=report['requires_review'])
        for key, name in report['files'].items():
            path = folder / name
            if path.is_absolute() or '..' in path.parts or not (root / path).is_file():
                raise ValueError(f'Invalid output reference in {folder}')
            record[key] = path.as_posix()
        groups[SPLITS[case['split']]].append(record)
        ids = {k: case[k] for k in ('split','patient','pair_id')}
        metrics = report['metrics']
        pairs.append({**ids, **{key: value for key,value in metrics.items() if key != 'targets'}})
        targets.extend({**ids, **t} for t in metrics['targets'])

    splits, stats = {}, []
    for split in [*SPLITS, 'all']:
        subset = [p for p in pairs if split == 'all' or p['split'] == split]
        if not subset:
            continue
        stats.extend(metric_rows(split, subset, [t for t in targets if split == 'all' or t['split'] == split]))
        if split in SPLITS:
            tre = patient_means(subset, 'target_tre_after_mm')
            splits[split] = dict(pairs=len(subset), patients=len({p['patient'] for p in subset}), target_patients=len(tre),
                                 patient_mean_target_tre_mm=mean(tre) if tre else None,
                                 mean_pair_dice=mean(p['dice_after'] for p in subset))
    checks = dict(state='verified', direction='MR_to_US', settings=config.to_dict(), splits=splits)
    with TemporaryDirectory(dir=root) as tmp:
        stage = Path(tmp)
        (stage/'metrics').mkdir()
        save_csv(stage/'metrics/pair_metrics.csv', pairs, pairs[0].keys())
        save_csv(stage/'metrics/target_metrics.csv', targets, ['split','patient','pair_id','target_number','before_mm','after_mm'])
        save_csv(stage/'metrics/metric_summary.csv', stats, stats[0].keys())
        (stage/'dataset.json').write_text(json.dumps(groups, indent=2, ensure_ascii=False),encoding='utf-8')
        (stage/'export_check.json').write_text(json.dumps(checks, indent=2),encoding='utf-8')
        (root/'metrics').mkdir(exist_ok=True)
        for table in (stage/'metrics').iterdir():
            table.replace(root/'metrics'/table.name)
        (stage/'export_check.json').replace(root/'export_check.json')
        (stage/'dataset.json').replace(root/'dataset.json')
