"""All public physical coordinates use LPS millimetres."""
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk

RAS_FROM_LPS = np.diag([-1., -1., 1., 1.])


def affine_lps(image):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    matrix[:3, 3] = image.GetOrigin()
    return matrix


def read_volume(value, *, assume_mm=False):
    image = sitk.Image(value) if isinstance(value, sitk.Image) else sitk.ReadImage(str(value))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Expected a scalar 3D volume, not a 4D or multichannel image')
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError('Volume contains non-finite pixel values')
    if any(not np.isfinite(v) or v <= 0 for v in image.GetSpacing()):
        raise ValueError('Spacing must be positive finite physical millimetres')
    a = affine_lps(image)
    if not np.isfinite(a).all():
        raise ValueError('Non-finite image geometry')
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError('Non-orthogonal image direction; check the source geometry')
    if not isinstance(value, sitk.Image) and str(value).lower().endswith(('.nii', '.nii.gz')):
        n = nib.load(str(value)); unit = n.header.get_xyzt_units()[0]
        scales = {'mm': 1., 'meter': 1000., 'micron': .001}
        if unit not in scales and not (unit == 'unknown' and assume_mm):
            raise ValueError('NIfTI spatial units are unknown. Correct the header, or explicitly use assume-mm when justified.')
        expected = n.affine.copy(); expected[:3] *= scales.get(unit, 1.)
        if not np.allclose(RAS_FROM_LPS @ a, expected, atol=2e-4, rtol=1e-6):
            raise ValueError('ITK geometry and NIfTI affine disagree; check qform/sform and units')
    return image


def binary_mask(value, *, assume_mm=False):
    image = read_volume(value, assume_mm=assume_mm)
    values = np.unique(sitk.GetArrayViewFromImage(image))
    if not np.isin(values, [0, 1]).all() or 1 not in values:
        raise ValueError('Expected a nonempty binary mask with values 0 and 1; extract the intended label first')
    return sitk.Cast(image, sitk.sitkUInt8)


def same_grid(a, b):
    return a.GetSize() == b.GetSize() and np.allclose(affine_lps(a), affine_lps(b), atol=1e-5, rtol=0)


def centroid(image):
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(sitk.Cast(image > 0, sitk.sitkUInt8))
    if not stats.HasLabel(1):
        raise ValueError('Empty mask')
    return np.asarray(stats.GetCentroid(1))


def apply_points(matrix, points):
    points = np.asarray(points, dtype=float)
    return points @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def itk_transform(matrix):
    tr = sitk.AffineTransform(3)
    tr.SetMatrix(np.asarray(matrix)[:3, :3].ravel().tolist())
    tr.SetTranslation(np.asarray(matrix)[:3, 3].tolist())
    return tr


def us_reference(us_mask, config):
    ref = sitk.Image(list(config.size), sitk.sitkFloat32)
    ref.SetSpacing(config.spacing_mm)
    ref.SetOrigin((centroid(us_mask) - (np.asarray(config.size) - 1) * np.asarray(config.spacing_mm) / 2).tolist())
    return ref


def resample_volume(image, reference, moving_to_fixed_lps=None, *, label=False):
    """ITK samples input using inverse(output point -> input point)."""
    inverse = np.linalg.inv(moving_to_fixed_lps) if moving_to_fixed_lps is not None else np.eye(4)
    return sitk.Resample(image, reference, itk_transform(inverse),
        sitk.sitkNearestNeighbor if label else sitk.sitkLinear, 0.,
        sitk.sitkUInt8 if label else sitk.sitkFloat32)


def write_nifti(image, path, reference):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)
    n = nib.load(str(path))
    if n.shape != reference.GetSize() or n.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Saved output has incorrect dimensions or spatial units')
    np.testing.assert_allclose(n.affine, RAS_FROM_LPS @ affine_lps(reference), atol=1e-5, rtol=0)

