"""Portable CSV manifests and resumable process-based batch execution."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import json
import multiprocessing
import re
import time
import traceback
import numpy as np
from .config import RegistrationConfig
from .pipeline import register_pair


def safe_id(value):
    if not value or value in ['.', '..'] or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        raise ValueError(f'Invalid identifier {value!r}: use letters, numbers, dots, underscores and hyphens')
    return value


def target_paths(row, modality, root):
    if row.get(modality + '_targets'):
        values = json.loads(row[modality + '_targets'])
        if not isinstance(values, dict):
            raise ValueError(f'{modality}_targets must be a JSON object mapping IDs to paths')
        return {int(n): resolve(root, p) for n, p in values.items()}
    result = {}
    for p in json.loads(row.get(modality + '_target_labels') or '[]'):
        match = re.search(r'target[_-]?(\d+)', Path(p).name, re.I)
        if not match:
            raise ValueError(f'Cannot infer Target ID from {p}; use {modality}_targets JSON mapping instead')
        number = int(match[1])
        if number in result:
            raise ValueError(f'Duplicate {modality} Target ID: {number}')
        result[number] = resolve(root, p)
    return result


def resolve(root, value):
    if not value:
        raise ValueError('Empty required volume path')
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def load_manifest(path, data_root=None):
    path = Path(path).resolve(); root = Path(data_root).resolve() if data_root else path.parent
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Manifest has no pairs')
    cases = []; seen = set()
    for row in rows:
        case_id = safe_id(row['pair_id'])
        if case_id in seen:
            raise ValueError(f'Duplicate pair_id: {case_id}')
        seen.add(case_id)
        patient = safe_id(row.get('patient') or case_id)
        split = row.get('split') or ''
        if split:
            safe_id(split)
        paths = {key: resolve(root, row[key]) for key in ['mr_image', 'us_image', 'mr_prostate_label', 'us_prostate_label']}
        warnings = [str(w) for m in ['mr', 'us'] for w in json.loads(row.get(m + '_warnings') or '[]')]
        cases.append(dict(pair_id=case_id, patient=patient, split=split, relative_output=str(Path(split) / patient / case_id),
            mr_image=paths['mr_image'], us_image=paths['us_image'], mr_prostate=paths['mr_prostate_label'], us_prostate=paths['us_prostate_label'],
            mr_targets=target_paths(row, 'mr', root), us_targets=target_paths(row, 'us', root), source_warnings=warnings, source_qc=row.get('status') or 'unspecified'))
    return cases


def _worker(case, output, config_dict, overwrite):
    if (Path(output) / 'STOP_REQUESTED').exists():
        return dict(pair_id=case['pair_id'], status='stopped')
    try:
        result = register_pair(case['mr_image'], case['us_image'], case['mr_prostate'], case['us_prostate'],
            Path(output) / case['relative_output'], mr_targets=case['mr_targets'], us_targets=case['us_targets'],
            case_id=case['pair_id'], source_warnings=case['source_warnings'],
            config=RegistrationConfig(**config_dict), overwrite=overwrite)
        r = result.report
        return dict(pair_id=case['pair_id'], patient=case['patient'], split=case['split'], status='success', resumed=result.resumed,
            relative_output=case['relative_output'], mr_image=case['relative_output'] + '/mr/image.nii.gz',
            us_image=case['relative_output'] + '/us/image.nii.gz',
            mr_prostate_label=case['relative_output'] + '/mr/prostate_label.nii.gz',
            us_prostate_label=case['relative_output'] + '/us/prostate_label.nii.gz',
            registration_json=case['relative_output'] + '/registration.json',
            dice=r['metrics']['dice_after'], target_tre_mm=r['metrics']['target_tre_after_mm'],
            rotation_degrees=r['diagnostics']['rotation_degrees'], requires_review=r['requires_review'], error='')
    except Exception as exc:
        return dict(pair_id=case['pair_id'], patient=case['patient'], split=case['split'], status='error',
                    relative_output=case['relative_output'], error=str(exc), traceback=traceback.format_exc())


def _write_state(output, results, expected, stopped=False):
    success = [r for r in results if r['status'] == 'success']
    status = 'running' if len(results) < expected else ('completed' if len(success) == expected else 'completed_with_errors')
    if stopped:
        status = 'stopped'
    progress = dict(state=status, total=expected, finished=len(results), successful=len(success), failed=len(results) - len(success))
    temp = output / 'progress.json.tmp'; temp.write_text(json.dumps(progress, indent=2)); temp.replace(output / 'progress.json')
    if success:
        temp = output / 'pairs.csv.tmp'
        with temp.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(success[0])); writer.writeheader(); writer.writerows(sorted(success, key=lambda r: r['pair_id']))
        temp.replace(output / 'pairs.csv')
    return progress


def run_batch(manifest, output_dir, *, data_root=None, config=None, workers=1, overwrite=False, monai=False):
    """Run a manifest; a bad pair is recorded without preventing other exports."""
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    config = config or RegistrationConfig(); cases = load_manifest(manifest, data_root)
    if monai:
        from .monai_export import validate_splits, write_export
        validate_splits(cases)
    output = Path(output_dir).resolve()
    if output in Path(manifest).resolve().parents:
        raise ValueError('Batch output must not contain the input manifest; use a separate output directory')
    output.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(str(output / '.batch.lock'), timeout=0):
        if monai:
            # A datalist advertises a complete export. Withdraw it during a retry.
            for name in ('dataset.json', 'export_check.json'):
                (output / name).unlink(missing_ok=True)
        results = []; _write_state(output, results, len(cases))
        if workers == 1:
            for case in cases:
                row = _worker(case, str(output), asdict(config), overwrite)
                if row['status'] == 'stopped':
                    break
                results.append(row)
                print(json.dumps({k: row.get(k) for k in ['pair_id', 'status', 'resumed', 'error']}, ensure_ascii=False), flush=True)
                _write_state(output, results, len(cases))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(_worker, case, str(output), asdict(config), overwrite): case for case in cases}
                for future in as_completed(futures):
                    case = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = dict(pair_id=case['pair_id'], status='error', error=str(exc))
                    if row['status'] == 'stopped':
                        continue
                    results.append(row); _write_state(output, results, len(cases))
                    print(json.dumps({k: row.get(k) for k in ['pair_id', 'status', 'resumed', 'error']}, ensure_ascii=False), flush=True)
        progress = _write_state(output, results, len(cases), (output / 'STOP_REQUESTED').exists()); successful = [r for r in results if r['status'] == 'success']
        patients = defaultdict(list)
        for r in successful:
            if r['target_tre_mm'] is not None:
                patients[r['patient']].append(r['target_tre_mm'])
        summary = dict(**progress, config=asdict(config), direction='MR_to_US',
            patient_equal_target_tre_mm=float(np.mean([np.mean(v) for v in patients.values()])) if patients else None,
            mean_pair_dice=float(np.mean([r['dice'] for r in successful])) if successful else None,
            review_flagged=sum(r['requires_review'] for r in successful),
            failures=[r for r in results if r['status'] != 'success'])
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        if monai and progress['state'] == 'completed':
            write_export(output, cases, successful, config)
        return summary

