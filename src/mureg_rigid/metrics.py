"""Evaluate saved MR-US labels with explicit, fixed metric definitions."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass, generate_binary_structure
from scipy.spatial import cKDTree

DEFINITIONS = {
    'TRE': 'Euclidean centroid distances in saved labels, paired nonempty channels 0-5; pooled across cases',
    'surface': 'Centers of foreground boundary voxels, 6-connected erosion, outside-image background; full affine world coordinates',
    'Chamfer': '0.5 * (mean nearest-neighbor surface distance A to B + mean B to A); unsquared Euclidean distance',
    'HD95': '95th percentile of concatenated A-to-B and B-to-A nearest-neighbor distances; numpy linear percentile',
    'Dice': '2*intersection/(foreground A + foreground B), channel0',
    'aggregation': 'TRE pooled over paired channels; other metrics mean across cases; population SD (ddof=0), also record sample SD',
    'units': 'Header-spacing-corrected world coordinates; source spatial units unspecified, millimeters not established',
}


def surface_points(mask, affine):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError('Empty prostate mask; cannot silently omit a case')
    boundary = mask & ~binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)
    return nib.affines.apply_affine(affine, np.argwhere(boundary))


def surface_metrics(a, b, affine):
    a, b = a > 0, b > 0
    pa, pb = surface_points(a, affine), surface_points(b, affine)
    ab = cKDTree(pb).query(pa, workers=1)[0]
    ba = cKDTree(pa).query(pb, workers=1)[0]
    return {
        'Chamfer': float(0.5 * (ab.mean() + ba.mean())),
        'HD95': float(np.percentile(np.concatenate([ab, ba]), 95)),
        'Dice': float(2 * np.count_nonzero(a & b) / (np.count_nonzero(a) + np.count_nonzero(b))),
    }


def label_metrics(a, b, affine):
    result = surface_metrics(a[..., 0], b[..., 0], affine)
    result['channel_errors'] = {}
    for c in range(6):
        if np.any(a[..., c]) and np.any(b[..., c]):
            pa, pb = [nib.affines.apply_affine(affine, center_of_mass(x[..., c] > 0)) for x in (a, b)]
            result['channel_errors'][str(c)] = float(np.linalg.norm(pa - pb))
    return result


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty or nonfinite metric')
    return {'mean': float(values.mean()), 'population_sd': float(values.std(ddof=0)),
            'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else None, 'count': len(values)}


def aggregate(rows):
    return {'TRE': stats([v for r in rows for v in r['channel_errors'].values()]),
            **{key: stats([r[key] for r in rows]) for key in ('Chamfer', 'Dice', 'HD95')}}


def evaluate_output(directory, split='test'):
    metas = json.loads((directory / split / 'meta/all_meta.json').read_text(encoding='utf-8'))
    if not all(m['accepted'] for m in metas):
        raise ValueError('Not every case was accepted; complete-cohort comparison required')
    rows, units = [], set()
    for m in metas:
        case = m['case']
        a, b = [nib.load(directory / split / f'{mod}_labels' / f'{case}.nii.gz') for mod in ('mr', 'us')]
        if a.shape != b.shape or not np.allclose(a.affine, b.affine):
            raise ValueError('Saved labels do not share a grid')
        units.update([a.header.get_xyzt_units()[0], b.header.get_xyzt_units()[0]])
        rows.append(dict(case=case, **label_metrics(np.asanyarray(a.dataobj), np.asanyarray(b.dataobj), a.affine)))
    return {'definitions': DEFINITIONS, 'source_output': str(directory.resolve()), 'split': split,
            'cases': len(rows), 'nifti_spatial_units': sorted(units), 'metrics': aggregate(rows), 'case_metrics': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--subset', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = evaluate_output(args.out_dir, args.subset)
    destination = args.report or args.out_dir / f'{args.subset}_four_metrics.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for name, metric in result['metrics'].items():
        print(f"{name}: {metric['mean']:.3f} +/- {metric['population_sd']:.3f} (n={metric['count']})", flush=True)


if __name__ == '__main__':
    main()
