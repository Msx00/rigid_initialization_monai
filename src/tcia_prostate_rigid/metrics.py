"""Explicit gland-overlap and Target-centroid metrics; distances are millimetres."""
import itertools
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt
from .geometry import affine_lps, apply_points, centroid, resample_volume


def dice(a, b):
    a = np.asarray(a) > 0; b = np.asarray(b) > 0
    denominator = int(a.sum() + b.sum())
    return float(2 * np.count_nonzero(a & b) / denominator) if denominator else None


def mask_bounds(image):
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(image)
    box = stats.GetBoundingBox(1)
    lo = np.asarray(box[:3], float) - .5
    hi = np.asarray(box[:3]) + np.asarray(box[3:]) - .5
    return apply_points(affine_lps(image), np.array(list(itertools.product(*zip(lo, hi)))))


def full_support_dice(moving, fixed, matrix, reference):
    """Use one grid phase, retaining whole glands before and after fitting."""
    mb = mask_bounds(moving); fb = mask_bounds(fixed); moved = apply_points(matrix, mb)
    origin = np.asarray(reference.GetOrigin()); spacing = np.asarray(reference.GetSpacing())
    indices = (np.vstack([mb, fb, moved]) - origin) / spacing
    lo = np.floor(indices.min(0)).astype(int) - 2; hi = np.ceil(indices.max(0)).astype(int) + 2
    if np.prod(hi - lo + 1, dtype=np.float64) > 128_000_000:
        raise ValueError('Evaluation grid exceeds 128 million voxels; check input millimetres and coordinate frames')
    grid = sitk.Image((hi - lo + 1).tolist(), sitk.sitkUInt8); grid.SetSpacing(spacing.tolist())
    grid.SetOrigin((origin + lo * spacing).tolist())
    fixed_array = sitk.GetArrayFromImage(resample_volume(fixed, grid, label=True))
    before = sitk.GetArrayFromImage(resample_volume(moving, grid, label=True))
    before_dice = dice(before, fixed_array); del before
    after = sitk.GetArrayFromImage(resample_volume(moving, grid, matrix, label=True))
    return before_dice, dice(after, fixed_array)


def measure(masks, output_masks, targets, matrix, reference):
    before, after = full_support_dice(masks['mr'], masks['us'], matrix, reference)
    mc = centroid(masks['mr']); fc = centroid(masks['us'])
    gland_before = float(np.linalg.norm(mc - fc)); gland_after = float(np.linalg.norm(apply_points(matrix, mc) - fc))
    distances = []
    for number in sorted(targets['mr'].keys() & targets['us'].keys()):
        a = centroid(targets['mr'][number]); b = centroid(targets['us'][number])
        distances.append(dict(target_number=number, before_mm=float(np.linalg.norm(a - b)),
                              after_mm=float(np.linalg.norm(apply_points(matrix, a) - b))))
    a = sitk.GetArrayViewFromImage(output_masks['mr']) > 0
    b = sitk.GetArrayViewFromImage(output_masks['us']) > 0
    sa = a ^ binary_erosion(a); sb = b ^ binary_erosion(b)
    surface = None; hd95 = None
    if sa.any() and sb.any():
        ab = distance_transform_edt(~sb, sampling=reference.GetSpacing()[::-1])[sa]
        ba = distance_transform_edt(~sa, sampling=reference.GetSpacing()[::-1])[sb]
        surface = float((ab.mean() + ba.mean()) / 2); hd95 = float(np.percentile(np.r_[ab, ba], 95))
    return dict(dice_before=before, dice_after=after, dice_after_output_grid=dice(a, b),
        gland_centroid_before_mm=gland_before, gland_centroid_after_mm=gland_after,
        target_count=len(distances), targets=distances,
        target_tre_before_mm=float(np.mean([v['before_mm'] for v in distances])) if distances else None,
        target_tre_after_mm=float(np.mean([v['after_mm'] for v in distances])) if distances else None,
        mixed_centroid_before_mm=float(np.mean([v['before_mm'] for v in distances] + [gland_before])),
        mixed_centroid_after_mm=float(np.mean([v['after_mm'] for v in distances] + [gland_after])),
        symmetric_surface_distance_output_grid_mm=surface, hd95_output_grid_mm=hd95)

