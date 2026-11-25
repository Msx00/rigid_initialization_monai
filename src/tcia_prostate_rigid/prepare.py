"""Build portable pairs.csv from official TCIA metadata/workbooks and STL archive."""
import csv
import json
from pathlib import Path
import random
import re
import math
import os
from collections import defaultdict
from xml.etree import ElementTree as ET
from zipfile import ZipFile
from .batch import safe_id


def workbook(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        table = []
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row', ns):
            values = {}
            for cell in row:
                key = re.sub(r'\d+', '', cell.get('r'))
                node = cell.find('s:v', ns)
                value = '' if node is None else node.text or ''
                if cell.get('t') == 's':
                    value = strings[int(value)]
                elif cell.get('t') == 'inlineStr':
                    value = ''.join(cell.find('s:is', ns).itertext())
                values[key] = value
            table.append(values)
    if not table:
        return []
    return [{header: row.get(key, '') for key, header in table[0].items()} for row in table[1:]]


def write_pairs(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def prepare_tcia(metadata, target_workbook, stls, output_dir, *, dicom_root=None, biopsy_workbook=None, mr_types=('T2',)):
    """Convert official pairs; retain failures/skips in prepare_report.json."""
    from .raw import convert_series
    metadata = Path(metadata).resolve(); output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    root = Path(dicom_root).resolve() if dicom_root else metadata.parent
    with metadata.open(encoding='utf-8-sig', newline='') as f:
        metadata_rows = list(csv.DictReader(f))
        meta = {r['Series UID']: r for r in metadata_rows}
    if len(meta) != len(metadata_rows):
        raise ValueError('Duplicate Series UID in metadata')
    paired = set()
    for r in workbook(target_workbook):
        paired.add((r['Patient ID'].strip(), r['seriesInstanceUID_MR'].strip(), r['seriesInstanceUID_US'].strip()))
    if biopsy_workbook:
        for r in workbook(biopsy_workbook):
            paired.add((r['Patient Number'].strip(), r['Series Instance UID (MRI)'].strip(), r['Series Instance UID (US)'].strip()))
    stls = Path(stls); archive = ZipFile(stls) if stls.is_file() and stls.suffix.lower() == '.zip' else None
    names = archive.namelist() if archive else [str(p) for p in stls.rglob('*') if p.suffix.lower() == '.stl']
    meshes = defaultdict(dict)
    for name in names:
        match = re.fullmatch(r'(.+?)-(ProstateSurface|Target\d+)-seriesUID-([\d.]+)\.STL', Path(name).name, re.I)
        if match:
            patient, kind, uid = match.groups(); key = 'prostate' if kind.lower() == 'prostatesurface' else f'target_{int(kind[6:])}'
            if key in meshes[uid]:
                raise ValueError(f'Duplicate STL for {uid}/{key}')
            meshes[uid][key] = (patient, name)
    converted = {}; rows = []; issues = []; per_patient = defaultdict(int)
    try:
        for patient, mr, us in sorted(paired):
            try:
                safe_id(patient)
                if mr not in meta or us not in meta:
                    raise ValueError('Missing metadata series')
                description = meta[mr].get('Series Description', '').lower()
                types = {kind.upper() for kind in mr_types}
                allowed = any(kind.lower() in description for kind in types)
                if not allowed or re.search(r'adc|dwi|diffusion|tracew', description):
                    issues.append(dict(patient=patient, mr_uid=mr, us_uid=us, status='skipped', reason='MR sequence filter')); continue
                for modality, uid in [('mr', mr), ('us', us)]:
                    if meta[uid]['Subject ID'] != patient or meta[uid]['Modality'].lower() != modality:
                        raise ValueError('Official pairing patient/modality mismatch')
                    if uid in converted:
                        continue
                    entries = meshes[uid]
                    if 'prostate' not in entries or any(p != patient for p, _ in entries.values()):
                        raise ValueError(f'Missing/mismatched prostate STL for {uid}')
                    def data(key):
                        name = entries[key][1]
                        return archive.read(name) if archive else Path(name).read_bytes()
                    relative = Path('series') / patient / modality / uid
                    location = meta[uid]['File Location'].replace('\\', '/').removeprefix('./')
                    source_dir = Path(location) if Path(location).is_absolute() else root / location
                    expected_files = int(meta[uid].get('Number of Images') or 0)
                    actual_files = sum(p.suffix.lower() == '.dcm' for p in source_dir.iterdir())
                    if expected_files and actual_files != expected_files:
                        raise ValueError(f'Incomplete DICOM series: metadata expects {expected_files} files, found {actual_files}')
                    target_ids = sorted(int(key.split('_')[1]) for key in entries if key.startswith('target_'))
                    report = convert_series(source_dir, data('prostate'), output / relative, modality=modality,
                        target_stls={n: data(f'target_{n}') for n in target_ids}, series_uid=uid)
                    if report['info']['patient'] != patient or report['info']['series_uid'] != uid:
                        raise ValueError('Actual DICOM patient/series does not match official metadata')
                    converted[uid] = (relative, target_ids, report)
                per_patient[patient] += 1
                row = dict(pair_id=f'{patient}_pair_{per_patient[patient]:02d}', patient=patient, split='', mr_uid=mr, us_uid=us)
                for modality, uid in [('mr', mr), ('us', us)]:
                    relative, numbers, report = converted[uid]
                    row[modality + '_image'] = (relative / 'image.nrrd').as_posix()
                    row[modality + '_prostate_label'] = (relative / 'prostate_label.nrrd').as_posix()
                    row[modality + '_target_labels'] = json.dumps([(relative / f'target_{n}_label.nrrd').as_posix() for n in numbers])
                    row[modality + '_warnings'] = json.dumps(report['warnings'])
                rows.append(row); write_pairs(output / 'pairs.csv', rows)
                print(json.dumps(dict(pair_id=row['pair_id'], status='prepared')), flush=True)
            except Exception as exc:
                issues.append(dict(patient=patient, mr_uid=mr, us_uid=us, status='error', reason=str(exc)))
    finally:
        if archive:
            archive.close()
    report = dict(prepared_pairs=len(rows), converted_series=len(converted), mr_types=list(mr_types),
                  issues=issues, status='completed' if rows and not any(i['status'] == 'error' for i in issues) else 'completed_with_errors')
    (output / 'prepare_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not rows:
        raise ValueError('No pairs were prepared; inspect prepare_report.json and sequence filter')
    return report


def split_manifest(manifest, output_dir, *, ratios=(.6, .1, .3), seed=42):
    """Patient-disjoint split manifests; source volumes are not copied/moved."""
    if len(ratios) != 3 or any(not math.isfinite(v) or v <= 0 for v in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError('Provide three positive ratios summing to 1')
    if Path(manifest).resolve() == (Path(output_dir) / 'pairs.csv').resolve():
        raise ValueError('Split output must not overwrite the input manifest')
    with Path(manifest).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    patients = sorted({r.get('patient') or r['pair_id'] for r in rows}); random.Random(seed).shuffle(patients)
    a = round(len(patients) * ratios[0]); b = a + round(len(patients) * ratios[1])
    groups = dict(train=set(patients[:a]), val=set(patients[a:b]), test=set(patients[b:]))
    if any(not v for v in groups.values()):
        raise ValueError('Too few patients for nonempty train/val/test splits')
    output = Path(output_dir); counts = {}; combined = []
    for split, ids in groups.items():
        selected = [{**r, 'split': split} for r in rows if (r.get('patient') or r['pair_id']) in ids]
        write_pairs(output / split / 'pairs.csv', selected); combined.extend(selected)
        counts[split] = dict(patients=len(ids), pairs=len(selected))
    write_pairs(output / 'pairs.csv', combined)
    report = dict(seed=seed, ratios=ratios, splits=counts, patient_disjoint=True,
                  data_root=Path(os.path.relpath(Path(manifest).resolve().parent, output.resolve())).as_posix(), note='Paths remain relative to original data root; pass --data-root to batch')
    (output / 'split.json').write_text(json.dumps(report, indent=2))
    return report

