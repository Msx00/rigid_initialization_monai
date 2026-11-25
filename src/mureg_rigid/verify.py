"""Verify outputs and evaluate native and resampled label-centroid TRE."""
import csv
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass

from . import run as registration

ROOT = None
SOURCE = None


def format_tre_values(value, *, as_text=False):
    """Round only exported TRE fields; keep full precision during evaluation.

    JSON retains numeric values; CSV/console use fixed three-decimal text.
    Missing measurements remain None, not zero.
    """
    if isinstance(value, dict):
        return {
            key: (f'{item:.3f}' if as_text else round(item, 3))
            if 'tre_' in key and isinstance(item, (int, float))
            else format_tre_values(item, as_text=as_text)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [format_tre_values(item, as_text=as_text) for item in value]
    return value


def evaluate_case(split, meta):
    case = meta['case']
    T = np.asarray(meta['T_us2mr_4x4'])
    angle = registration.rotation_angle_deg(T[:3, :3])
    assert angle <= registration.ICP_MAX_ROT_DEG + 1e-7, (split, case, angle)
    assert meta['icp_params']['max_rot_deg'] == registration.ICP_MAX_ROT_DEG
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-6)
    assert np.allclose(T @ np.asarray(meta['T_mr2us_4x4']), np.eye(4), atol=1e-6)
    saved = {}
    for folder in ['mr_images', 'us_images', 'mr_labels', 'us_labels', 'mr_labels_maps', 'us_labels_maps', 'us_images_before_rigid']:
        image = nib.load(ROOT / split / folder / (case + '.nii.gz'))
        data = np.asanyarray(image.dataobj)
        assert image.shape[:3] == (128, 128, 128)
        assert np.isfinite(data).all(), (split, case, folder)
        assert np.allclose(image.header.get_zooms()[:3], np.linalg.norm(image.affine[:3, :3], axis=0))
        if folder in ['mr_labels', 'us_labels']:
            assert image.shape[-1] == 6 and set(np.unique(data)).issubset({0, 255})
            saved[folder] = (image, data)
        if folder == 'mr_images':
            reference = image
        elif folder != 'us_images_before_rigid':
            assert np.allclose(image.affine, reference.affine)
    mr_image, mr_data = saved['mr_labels']
    us_image, us_data = saved['us_labels']
    dice = registration.dice_binary_np(mr_data[..., 0], us_data[..., 0])
    assert np.isclose(dice, meta['label0_dice'])

    native = [nib.load(SOURCE / split / (mod + '_labels') / (case + '.nii.gz')) for mod in ['us', 'mr']]
    arrays = [np.asanyarray(image.dataobj) for image in native]
    expected_channels = {c for c in range(6) if all(np.any(a[..., c]) for a in arrays)}
    actual_channels = {c for c in range(6) if np.any(mr_data[..., c]) and np.any(us_data[..., c])}
    assert actual_channels == expected_channels, (split, case, 'valid paired label channels changed', expected_channels, actual_channels)
    affines = [registration.affine_fix_spacing(image.affine, image.header.get_zooms()[:3]) for image in native]
    before, after, saved_errors = [], [], []
    for c in range(1, 6):
        if any(not np.any(data[..., c]) for data in arrays):
            continue
        points = [affine @ np.r_[center_of_mass(data[..., c] > 0), 1]
                  for data, affine in zip(arrays, affines)]
        before.append(float(np.linalg.norm((points[0] - points[1])[:3])))
        after.append(float(np.linalg.norm((T @ points[0] - points[1])[:3])))
        if np.any(mr_data[..., c]) and np.any(us_data[..., c]):
            a = mr_image.affine @ np.r_[center_of_mass(mr_data[..., c] > 0), 1]
            b = us_image.affine @ np.r_[center_of_mass(us_data[..., c] > 0), 1]
            saved_errors.append(float(np.linalg.norm((a - b)[:3])))
    saved_channels = {}
    for c in range(6):
        error = None
        if np.any(mr_data[..., c]) and np.any(us_data[..., c]):
            a = mr_image.affine @ np.r_[center_of_mass(mr_data[..., c] > 0), 1]
            b = us_image.affine @ np.r_[center_of_mass(us_data[..., c] > 0), 1]
            error = float(np.linalg.norm((a - b)[:3]))
        saved_channels[f'tre_saved_channel_{c}'] = error
    return {'subset': split, 'case': case, 'dice_ch0': dice, 'rotation_degrees': angle,
            'attempt': meta['best_attempt'], 'valid_landmarks': len(after),
            'tre_before': float(np.mean(before)) if before else None,
            'tre_after': float(np.mean(after)) if after else None,
            'saved_label_centroid_tre_after': float(np.mean(saved_errors)) if saved_errors else None,
            **saved_channels}


def main():
    global ROOT, SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--subsets', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    args = parser.parse_args()
    ROOT, SOURCE = args.out_dir.resolve(), args.data_root.resolve()
    run_config = json.loads((ROOT / 'run_config.json').read_text(encoding='utf-8'))
    registration.configure_rotation_limit(run_config['max_rotation_degrees'])
    registration.ICP_MAX_ROT_DEG = registration.registration_parameters()['max_rot_deg']
    summaries, rows, rejected = {}, [], []
    for split in args.subsets:
        summary = json.loads((ROOT / split / 'meta' / 'summary.json').read_text(encoding='utf-8'))
        all_meta = json.loads((ROOT / split / 'meta' / 'all_meta.json').read_text(encoding='utf-8'))
        assert len(all_meta) == summary['num_total']
        accepted = [m for m in all_meta if m['accepted']]
        assert len(accepted) == summary['num_accepted']
        expected = {m['case'] + '.nii.gz' for m in accepted}
        for folder in ['mr_images', 'us_images', 'mr_labels', 'us_labels', 'mr_labels_maps', 'us_labels_maps', 'us_images_before_rigid']:
            assert {p.name for p in (ROOT / split / folder).glob('*.nii.gz')} == expected
        for meta in all_meta:
            if meta['accepted']:
                rows.append(evaluate_case(split, meta))
            else:
                rejected.append({'subset': split, 'case': meta['case'], 'reason': meta['failure_reason'],
                                 'best_dice': meta.get('best_dice'), 'best_rotation': meta.get('best_rot_angle_deg')})
        group = [r for r in rows if r['subset'] == split]
        summaries[split] = {'total': summary['num_total'], 'accepted': len(group),
                            'skipped': summary['num_skipped'],
                            'mean_dice': float(np.mean([r['dice_ch0'] for r in group])) if group else None,
                            'max_rotation_degrees': max((r['rotation_degrees'] for r in group), default=None),
                            'mean_case_tre_before': float(np.mean([r['tre_before'] for r in group if r['tre_before'] is not None])) if any(r['tre_before'] is not None for r in group) else None,
                            'mean_case_tre_after': float(np.mean([r['tre_after'] for r in group if r['tre_after'] is not None])) if any(r['tre_after'] is not None for r in group) else None}
        pooled = [r[f'tre_saved_channel_{c}'] for r in group for c in range(6)
                  if r[f'tre_saved_channel_{c}'] is not None]
        summaries[split].update({
            'valid_saved_label0_5_pairs': len(pooled),
            'tre_saved_pooled_labels0_5_mean': float(np.mean(pooled)) if pooled else None,
        })
        print(split, format_tre_values(summaries[split], as_text=True), flush=True)
    config_path = ROOT / 'run_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    report = {'verification': 'passed', 'tre_definition': 'Channels 1-5 original mask centroids; forward transform; mean within each case, then across cases',
              'saved_pooled_tre_definition': 'Pool all nonempty matching channel 0-5 centroid distances from saved/resampled labels across cases; label0 also participates in registration',
              'tuning_dataset': config.get('registration_parameters', {}).get('tuning_dataset'),
              'distance_units': 'Header-spacing-corrected world units; NIfTI unit is unknown',
              'subsets': summaries, 'rejected_cases': rejected,
              'verified_nifti_files': len(rows) * 7}
    (ROOT / 'verification_and_tre.json').write_text(json.dumps(format_tre_values(report), indent=2), encoding='utf-8')
    with (ROOT / 'case_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['subset', 'case'])
        writer.writeheader()
        writer.writerows(format_tre_values(rows, as_text=True))
    print(json.dumps(format_tre_values(report, as_text=True), indent=2))


if __name__ == '__main__':
    main()
