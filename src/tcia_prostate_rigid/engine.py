"""Stateless adaptation of the validated regularized-PCA + upstream rigid ICP."""
from dataclasses import dataclass
import itertools
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import vtk
from scipy.spatial.transform import Rotation
from .config import RegistrationConfig
from .geometry import RAS_FROM_LPS, affine_lps, binary_mask
from ._vendor.pca_icp.io_vtk import load_binary_segmentation_nii_as_polydata
from ._vendor.pca_icp.preprocess import clean_and_decimate
from ._vendor.pca_icp.pca_init import compute_pca_frame
from ._vendor.pca_icp.transforms import apply_matrix_to_polydata
from ._vendor.pca_icp.icp import run_icp


@dataclass
class RigidFit:
    matrix_lps: np.ndarray
    matrix_ras: np.ndarray
    diagnostics: dict


def validate_rigid(matrix):
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid rigid transform')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6):
        raise ValueError('Transform contains scaling, shear or reflection')


def fit_rigid(moving_mask, fixed_mask, config=None):
    """Fit source -> target using binary masks in their native mm geometry.

    Defaults mean MR moving -> US fixed. No Target landmarks are accepted here.
    """
    config = config or RegistrationConfig()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(config.threads)
    vtk.vtkMultiThreader.SetGlobalDefaultNumberOfThreads(config.threads)
    vtk.vtkSMPTools.Initialize(config.threads)
    masks = [binary_mask(v, assume_mm=config.assume_unknown_nifti_units_are_mm) for v in [moving_mask, fixed_mask]]
    meshes = []
    with tempfile.TemporaryDirectory(prefix='tcia-rigid-') as temp:
        for i, mask in enumerate(masks):
            path = Path(temp) / f'mask{i}.nii.gz'; sitk.WriteImage(mask, str(path))
            # Keep the same NIfTI-affine precision and extraction path as the validated run.
            import nibabel as nib
            np.testing.assert_allclose(nib.load(path).affine, RAS_FROM_LPS @ affine_lps(mask), atol=1e-5, rtol=0)
            meshes.append(clean_and_decimate(load_binary_segmentation_nii_as_polydata(str(path), 1, config.smoothing), config.decimation_reduction))
    moving, fixed = [compute_pca_frame(poly) for poly in meshes]
    candidates = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            correspondence = np.eye(3)[:, permutation] @ np.diag(signs)
            rotation = fixed.eigenvectors @ correspondence @ moving.eigenvectors.T
            if np.linalg.det(rotation) < 0:
                continue
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
            candidates.append((angle, rotation, permutation, signs))
    full_angle, pca, permutation, signs = min(candidates, key=lambda item: item[0])
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_rotvec(config.pca_rotation_gain * Rotation.from_matrix(pca).as_rotvec()).as_matrix()
    initial[:3, 3] = fixed.centroid - initial[:3, :3] @ moving.centroid
    refined = run_icp(apply_matrix_to_polydata(meshes[0], initial), meshes[1], mode='rigid',
                      max_iterations=config.iterations, max_landmarks=config.landmarks, max_mean_distance=1e-3)
    ras = refined.matrix4x4 @ initial; validate_rigid(ras)
    lps = RAS_FROM_LPS @ ras @ RAS_FROM_LPS
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(ras[:3, :3]) - 1) / 2, -1, 1))))
    diagnostics = dict(rotation_degrees=angle, rotation_warning=angle > config.rotation_warning_degrees,
        iterations_used=int(refined.iterations), iteration_limit_reached=int(refined.iterations) == config.iterations,
        last_mean_update_mm=float(refined.mean_distance), pca_full_rotation_degrees=full_angle,
        pca_axis_permutation=list(permutation), pca_axis_signs=list(signs),
        initial_matrix_ras=initial.tolist(), icp_matrix_ras=refined.matrix4x4.tolist(),
        mesh_points=dict(moving=int(meshes[0].GetNumberOfPoints()), fixed=int(meshes[1].GetNumberOfPoints())),
        superior_axis_dot=float(lps[2, 2]))
    return RigidFit(matrix_lps=lps, matrix_ras=ras, diagnostics=diagnostics)

