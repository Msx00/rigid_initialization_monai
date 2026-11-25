import dataclasses
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation
from tcia_prostate_rigid import RegistrationConfig, fit_rigid
from tcia_prostate_rigid.geometry import centroid, resample_volume, read_volume, binary_mask


class GeometryTests(unittest.TestCase):
    def test_configuration_rejects_invalid_physical_settings(self):
        for kwargs in [dict(spacing_mm=(0, .8, .8)), dict(size=(1, 175, 175)), dict(pca_rotation_gain=2),
                       dict(iterations=0), dict(landmarks=-1), dict(smoothing=-1), dict(threads=0),
                       dict(rotation_warning_degrees=float('nan')), dict(decimation_reduction=1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RegistrationConfig(**kwargs)

    def fixture(self):
        a = np.zeros((32, 32, 32), np.uint8); a[10:17, 9:14, 7:10] = 1
        im = sitk.GetImageFromArray(a); im.SetSpacing((.9, 1.3, 1.9)); im.SetOrigin((32., -40., 11.))
        direction = Rotation.from_euler('zyx', [17, -13, 9], degrees=True).as_matrix()
        im.SetDirection(direction.ravel().tolist())
        T = np.eye(4); T[:3, :3] = Rotation.from_euler('zyx', [25, 8, -11], degrees=True).as_matrix(); T[:3, 3] = [23, -14, 6]
        c = centroid(im); fixed = T[:3, :3] @ c + T[:3, 3]
        ref = sitk.Image([31] * 3, sitk.sitkFloat32); ref.SetSpacing([.8] * 3); ref.SetOrigin((fixed - 12).tolist())
        return im, T, ref, c, fixed

    def test_oblique_anisotropic_mask_forward_direction(self):
        im, T, ref, c, fixed = self.fixture()
        output = resample_volume(im, ref, T, label=True)
        self.assertLess(np.linalg.norm(centroid(output) - fixed), .5)
        self.assertEqual(set(np.unique(sitk.GetArrayViewFromImage(output))), {0, 1})

    def test_oblique_intensity_ramp_uses_inverse_sampling(self):
        im, T, ref, c, _ = self.fixture()
        z, y, x = np.indices((32, 32, 32)); ijk = np.stack([x, y, z], axis=-1)
        from tcia_prostate_rigid.geometry import affine_lps
        a = affine_lps(im); points = ijk @ a[:3, :3].T + a[:3, 3]; weights = np.array([.2, .3, .4])
        ramp = sitk.GetImageFromArray((points @ weights).astype(np.float32)); ramp.CopyInformation(im)
        output = resample_volume(ramp, ref, T)
        self.assertAlmostEqual(output.GetPixel(15, 15, 15), float(c @ weights), places=4)

    def test_unknown_nifti_units_require_explicit_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'unknown.nii.gz'; nib.save(nib.Nifti1Image(np.ones((4, 5, 6), np.uint8), np.eye(4)), p)
            with self.assertRaisesRegex(ValueError, 'units'):
                read_volume(p)
            self.assertEqual(read_volume(p, assume_mm=True).GetSize(), (4, 5, 6))

    def test_binary_mask_rejects_empty_and_multilabel(self):
        for a in [np.zeros((5, 5, 5), np.uint8), np.full((5, 5, 5), 2, np.uint8)]:
            with self.assertRaises(ValueError):
                binary_mask(sitk.GetImageFromArray(a))

    def test_nonfinite_image_is_rejected(self):
        a = np.ones((5, 5, 5), np.float32); a[1, 2, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            read_volume(sitk.GetImageFromArray(a))

    def test_nifti_metre_units_are_converted_to_mm(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metres.nii.gz'
            n = nib.Nifti1Image(np.ones((4, 5, 6), np.uint8), np.diag([.001, .002, .003, 1]))
            n.header.set_xyzt_units('meter'); nib.save(n, path)
            im = read_volume(path)
            np.testing.assert_allclose(im.GetSpacing(), [1, 2, 3], atol=1e-6)

    def test_fit_known_mm_translation_in_oblique_grid(self):
        im, _, _, _, _ = self.fixture(); fixed = sitk.Image(im); shift = np.array([8., -6., 4.])
        fixed.SetOrigin((np.array(im.GetOrigin()) + shift).tolist())
        fit = fit_rigid(im, fixed, RegistrationConfig(iterations=20))
        np.testing.assert_allclose(fit.matrix_lps[:3, :3], np.eye(3), atol=1e-4)
        # Upstream ICP stops at 1e-3 mm updates and uses float mesh vertices.
        # Check physical point errors rather than exact floating matrix entries.
        for p in [centroid(im), np.array(im.GetOrigin())]:
            actual = fit.matrix_lps[:3, :3] @ p + fit.matrix_lps[:3, 3]
            self.assertLess(np.linalg.norm(actual - (p + shift)), .01)
        self.assertAlmostEqual(np.linalg.det(fit.matrix_lps[:3, :3]), 1, places=6)

