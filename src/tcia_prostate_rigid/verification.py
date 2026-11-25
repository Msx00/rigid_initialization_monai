"""Independent checks of written NIfTI geometry and physical interpolation."""
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from .geometry import RAS_FROM_LPS, affine_lps, apply_points


def verify_outputs(folder, files, native, native_masks, reference, matrix):
    forward = sitk.ReadTransform(str(folder / files['forward_transform']))
    backward = sitk.ReadTransform(str(folder / files['inverse_transform']))
    for name in files.values():
        if not name.endswith('.nii.gz'):
            continue
        n = nib.load(folder / name)
        assert n.shape == reference.GetSize() and n.header.get_xyzt_units()[0] == 'mm'
        np.testing.assert_allclose(n.affine, RAS_FROM_LPS @ affine_lps(reference), atol=1e-5, rtol=0)
    rng = np.random.default_rng(42)
    for point in rng.uniform(-100, 100, (10, 3)):
        np.testing.assert_allclose(forward.TransformPoint(point.tolist()), apply_points(matrix, point), atol=1e-6)
        np.testing.assert_allclose(backward.TransformPoint(forward.TransformPoint(point.tolist())), point, atol=1e-6)
    fixed = sitk.ReadImage(str(folder / files['us_prostate_label']))
    foreground = np.argwhere(sitk.GetArrayViewFromImage(fixed) > 0)
    if not len(foreground):
        raise ValueError('US gland disappeared from output grid')
    ijk = np.vstack([rng.integers(0, reference.GetSize(), (300, 3)),
                    foreground[rng.integers(0, len(foreground), 300), ::-1]])
    world = [reference.TransformIndexToPhysicalPoint(tuple(map(int, p))) for p in ijk]
    checks = {}
    for mod in ['mr', 'us']:
        points = [backward.TransformPoint(p) for p in world] if mod == 'mr' else world
        positions = np.asarray([native[mod].TransformPhysicalPointToContinuousIndex(p) for p in points])
        valid = ((positions >= 1) & (positions < np.asarray(native[mod].GetSize()) - 2)).all(1)
        if valid.sum() < 10:
            raise ValueError(f'{mod}: insufficient valid image samples; inspect physical geometry')
        source = sitk.GetArrayViewFromImage(native[mod])
        expected = map_coordinates(source, positions[valid, ::-1].T, order=1, output=np.float64, prefilter=False)
        output = sitk.ReadImage(str(folder / files[mod + '_image']))
        actual = sitk.GetArrayViewFromImage(output)[tuple(ijk[valid, ::-1].T)]
        scale = max(float(np.max(np.abs(source.astype(np.float64)))), 1.)
        relative_error = float(np.max(np.abs(expected - actual))) / scale
        if not np.isfinite(relative_error) or relative_error >= 5e-5:
            raise ValueError(f'{mod}: independent image sampling failed: {relative_error}')
        no_tie = (np.abs(positions[valid] - np.floor(positions[valid]) - .5) > 1e-4).all(1)
        xyz = np.floor(positions[valid][no_tie] + .5).astype(int)
        wanted = sitk.GetArrayViewFromImage(native_masks[mod])[tuple(xyz[:, ::-1].T)] > 0
        label = sitk.ReadImage(str(folder / files[mod + '_prostate_label']))
        found = sitk.GetArrayViewFromImage(label)[tuple(ijk[valid][no_tie, ::-1].T)] > 0
        if not np.array_equal(wanted, found):
            raise ValueError(f'{mod}: independent nearest-neighbor label sampling failed')
        checks[mod] = dict(samples=int(valid.sum()), relative_image_error=relative_error, nearest_neighbor_matches=True)
    return dict(status='passed', saved_grid_and_mm_affine=True, forward_inverse_roundtrip=True,
                independent_sampling=checks, anatomical_accuracy_guaranteed=False)

