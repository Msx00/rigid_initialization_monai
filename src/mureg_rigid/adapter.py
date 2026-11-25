"""Regularized PCA initialization followed by the original author's rigid ICP.

Upstream: https://github.com/dalbenzioG/Initial_Registration
The PCA frames, mesh extraction and ICP are actual upstream calls. Axis-order
disambiguation and rotation damping are adapter additions, not vanilla PCA.
The packaged default profile was selected on train + validation; see docs/METHOD.md for development provenance.
"""
from dataclasses import asdict
import itertools
import numpy as np
from scipy.spatial.transform import Rotation

from pca_icp import icp
from pca_icp.io_vtk import load_binary_segmentation_nii_as_polydata
from pca_icp.pca_init import compute_pca_frame
from pca_icp.preprocess import clean_and_decimate
from pca_icp.transforms import apply_matrix_to_polydata, compose_matrices
from pca_icp.types import RegistrationConfig

_SETTINGS = {'smoothing': 0, 'landmarks': 7000, 'iterations': 30, 'pca_rotation_gain': 0.10, 'decimation_reduction': 0.2}


_MAX_ROTATION_DEGREES = 30.0
_SELECTION_DATASET = 'train_val'


def configure_selection_dataset(dataset):
    global _SELECTION_DATASET
    if dataset not in ['unspecified', 'train', 'val', 'train_val', 'test']:
        raise ValueError('Selection dataset must be unspecified, train, val, train_val or test')
    _SELECTION_DATASET = dataset


def configure_rotation_limit(degrees):
    """Set the post-registration acceptance gate; does not change optimization."""
    global _MAX_ROTATION_DEGREES
    degrees = float(degrees)
    if not np.isfinite(degrees) or not 0 <= degrees <= 180:
        raise ValueError('Rotation limit must be finite and between 0 and 180 degrees')
    _MAX_ROTATION_DEGREES = degrees


def configure_registration(settings):
    """Load a single global configuration; never select settings per case."""
    unknown = set(settings) - set(_SETTINGS)
    if unknown:
        raise ValueError(f'Unknown registration settings: {sorted(unknown)}')
    merged = dict(_SETTINGS)
    # Profiles predating mesh simplification always used zero reduction.
    merged['decimation_reduction'] = 0.0
    merged.update(settings)
    for key in ['smoothing', 'landmarks', 'iterations']:
        if type(merged[key]) is not int or merged[key] < (0 if key == 'smoothing' else 1):
            raise ValueError(f'Invalid {key}')
    if not 0 <= merged['pca_rotation_gain'] <= 1:
        raise ValueError('PCA rotation gain must be between 0 and 1 inclusive')
    if not 0 <= merged['decimation_reduction'] < 1:
        raise ValueError('Mesh reduction must be at least 0 and less than 1')
    _SETTINGS.update(merged)


def registration_parameters():
    return {
        'upstream_config': asdict(RegistrationConfig(icp_mode='rigid',
            smoothing_iterations=_SETTINGS['smoothing'], icp_max_landmarks=_SETTINGS['landmarks'],
            icp_max_iterations=_SETTINGS['iterations'], decimation_reduction=_SETTINGS['decimation_reduction'],
            run_multistart_on_unstable_pca=False)),
        'method': ('centroid_alignment_then_author_icp' if _SETTINGS['pca_rotation_gain'] == 0
                   else 'regularized_pca_axis_permutations_then_author_icp'),
        'max_rot_deg': _MAX_ROTATION_DEGREES,
        'pca_rotation_gain': _SETTINGS['pca_rotation_gain'],
        'pca_axis_candidates': 24,
        'candidate_selection': 'minimum_rotation_angle',
        'icp_multistart': False,
        'initialization': 'centroid_only' if _SETTINGS['pca_rotation_gain'] == 0 else 'regularized_pca_only',
        'primary_backend': 'pca_icp.icp.run_icp',
        'pca_backend': 'pca_icp.pca_init.compute_pca_frame',
        'acceptance': 'valid_rigid_transform_within_rotation_limit',
        'tuning_dataset': _SELECTION_DATASET,
        'tuning_objective': 'See experiment report for the selection objective and cohort',
    }


def rotation_angle_deg(rotation):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def _validate_rigid(matrix):
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid transformation matrix')
    rotation = matrix[:3, :3]
    if not (np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError('Transformation is not rigid')


def register_masks(source_path, target_path):
    """Use only the two channel-0 masks; no landmark labels or TRE are inputs."""
    params = registration_parameters()
    config = RegistrationConfig(**params['upstream_config'])
    source, target = [clean_and_decimate(load_binary_segmentation_nii_as_polydata(
        str(path), label_value=label, smoothing_iterations=config.smoothing_iterations),
        config.decimation_reduction) for path, label in
        [(source_path, config.source_label), (target_path, config.target_label)]]
    return register_meshes(source, target)


def register_meshes(source, target):
    """Same registration path with preloaded meshes for reproducible parameter sweeps."""
    params = registration_parameters()
    config = RegistrationConfig(**params['upstream_config'])
    moving, fixed = compute_pca_frame(source), compute_pca_frame(target)
    candidates = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            correspondence = np.eye(3)[:, permutation] @ np.diag(signs)
            rotation = fixed.eigenvectors @ correspondence @ moving.eigenvectors.T
            if np.linalg.det(rotation) < 0:
                continue
            candidates.append((rotation_angle_deg(rotation), rotation, permutation, signs))
    angle, pca_rotation, permutation, signs = min(candidates, key=lambda c: c[0])
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_rotvec(
        params['pca_rotation_gain'] * Rotation.from_matrix(pca_rotation).as_rotvec()).as_matrix()
    initial[:3, 3] = fixed.centroid - initial[:3, :3] @ moving.centroid
    refined = icp.run_icp(apply_matrix_to_polydata(source, initial), target,
                         mode=config.icp_mode, max_iterations=config.icp_max_iterations,
                         max_landmarks=config.icp_max_landmarks,
                         max_mean_distance=config.icp_max_mean_distance)
    matrix = compose_matrices(initial, refined.matrix4x4)
    _validate_rigid(matrix)
    final_angle = rotation_angle_deg(matrix[:3, :3])
    accepted = final_angle <= params['max_rot_deg']
    attempt = {
        'attempt_index': 1, 'init_type': 'centroid_only' if params['pca_rotation_gain'] == 0 else 'regularized_pca',
        'backend': params['primary_backend'], 'pca_backend': params['pca_backend'],
        'pca_axis_permutation': list(permutation), 'pca_axis_signs': list(signs),
        'full_pca_rotation_degrees': angle, 'pca_rotation_gain': params['pca_rotation_gain'],
        'pca_frames': {
            name: {'centroid': frame.centroid.tolist(),
                   'eigenvalues': frame.eigenvalues.tolist(),
                   'eigenvectors_columns': frame.eigenvectors.tolist(),
                   'basis_determinant': float(np.linalg.det(frame.eigenvectors))}
            for name, frame in [('moving_us', moving), ('fixed_mr', fixed)]
        },
        'initial_matrix': initial.tolist(), 'icp_matrix': refined.matrix4x4.tolist(),
        'T_mv2fx_4x4': matrix.tolist(), 'rot_angle_deg': final_angle,
        'iterations': int(refined.iterations), 'vtk_mean_update_distance': float(refined.mean_distance),
        'accepted': bool(accepted),
    }
    return {
        'accepted': bool(accepted),
        'failure_reason': None if accepted else 'rotation_threshold_not_met',
        'T_mv2fx': matrix, 'best_rot_angle_deg': final_angle,
        'best_attempt': 1, 'attempts': [attempt], 'backend': params['primary_backend'],
        'mesh_points': {'source': source.GetNumberOfPoints(), 'target': target.GetNumberOfPoints()},
    }
