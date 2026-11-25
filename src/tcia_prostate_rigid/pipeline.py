"""One-call registration with verified, atomic and resumable case outputs."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import uuid
import numpy as np
import SimpleITK as sitk
from filelock import FileLock
from . import __version__
from .config import RegistrationConfig
from .engine import fit_rigid
from .geometry import (read_volume, binary_mask, same_grid, us_reference, resample_volume,
                       write_nifti, itk_transform, apply_points, affine_lps)
from .metrics import measure, mask_bounds
from .verification import verify_outputs


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def fingerprint(value):
    if not isinstance(value, sitk.Image):
        return file_hash(value)
    h = hashlib.sha256(sitk.GetArrayViewFromImage(value).tobytes())
    h.update(affine_lps(value).tobytes())
    return h.hexdigest()


def implementation_fingerprint():
    folder = Path(__file__).parent
    files = [folder / name for name in ['config.py', 'engine.py', 'geometry.py', 'metrics.py', 'pipeline.py', 'verification.py']]
    files += sorted((folder / '_vendor').rglob('*.py'))
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(folder)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


@dataclass
class RegistrationResult:
    output_dir: Path
    report: dict
    resumed: bool = False

    @property
    def matrix_lps(self):
        return np.asarray(self.report['transform']['mr_to_us_lps_mm'])

    @property
    def metrics(self):
        return self.report['metrics']

    @property
    def paths(self):
        return {key: self.output_dir / value for key, value in self.report['files'].items()}


def register_pair(mr_image, us_image, mr_prostate, us_prostate, output_dir, *,
                  mr_targets=None, us_targets=None, config=None, overwrite=False,
                  case_id=None, source_warnings=None):
    """Fit MR -> US and save co-gridded images, masks, Targets and transforms.

    Inputs are paths or SimpleITK.Image objects in physical millimetres. Target
    dictionaries map positive integer IDs to binary-mask paths/images. Existing
    successful outputs resume only when content and configuration match.
    """
    config = config or RegistrationConfig(); output = Path(output_dir).resolve()
    targets_in = {'mr': dict(mr_targets or {}), 'us': dict(us_targets or {})}
    inputs = dict(mr_image=mr_image, us_image=us_image, mr_prostate=mr_prostate, us_prostate=us_prostate)
    for mod, values in targets_in.items():
        if any(type(k) is not int or not 1 <= k <= 65535 for k in values):
            raise ValueError('Target IDs must be positive integers <= 65535')
        for number, value in values.items():
            inputs[f'{mod}_target_{number}'] = value
    for value in inputs.values():
        if not isinstance(value, sitk.Image):
            source = Path(value).resolve()
            if output == source or output in source.parents:
                raise ValueError('Output directory must not contain any input file')
    identity = dict(version=__version__, implementation_sha256=implementation_fingerprint(), config=config.to_dict(), inputs={k: fingerprint(v) for k, v in sorted(inputs.items())},
        case_id=case_id, source_warnings=source_warnings or [],
        versions={k: importlib.metadata.version(k) for k in ['numpy', 'scipy', 'vtk', 'nibabel', 'SimpleITK', 'scikit-image']})
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / f'.{output.name}.lock'), timeout=0):
        old = output / 'registration.json'
        if output.exists() and any(output.iterdir()) and not old.exists():
            raise FileExistsError('Nonempty output is not a registered case; choose a separate case directory')
        if old.exists() and not overwrite:
            report = json.loads(old.read_text())
            if report.get('input_signature') != signature:
                raise FileExistsError('Existing output has different inputs/settings; use another directory or overwrite=True')
            if report.get('status') != 'success':
                raise FileExistsError('Existing output is incomplete; use overwrite=True to replace it')
            for name, checksum in report['output_sha256'].items():
                if not (output / name).is_file() or file_hash(output / name) != checksum:
                    raise ValueError('Existing output is missing or changed; use overwrite=True to rebuild it')
            return RegistrationResult(output, report, resumed=True)
        if output.exists() and any(output.iterdir()) and not overwrite:
            raise FileExistsError('Output directory is nonempty; use a fresh case directory or overwrite=True')
        native = {m: read_volume(inputs[m + '_image'], assume_mm=config.assume_unknown_nifti_units_are_mm) for m in ['mr', 'us']}
        masks = {m: binary_mask(inputs[m + '_prostate'], assume_mm=config.assume_unknown_nifti_units_are_mm) for m in ['mr', 'us']}
        targets = {m: {n: binary_mask(v, assume_mm=config.assume_unknown_nifti_units_are_mm) for n, v in targets_in[m].items()} for m in ['mr', 'us']}
        for mod in ['mr', 'us']:
            if not same_grid(native[mod], masks[mod]) or any(not same_grid(native[mod], t) for t in targets[mod].values()):
                raise ValueError(f'{mod}: image and masks must share their original physical grid')
        fit = fit_rigid(masks['mr'], masks['us'], config)
        reference = us_reference(masks['us'], config)
        stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
        files = {}; warnings = list(source_warnings or []); label_stats = []; output_masks = {}
        try:
            for mod, transform in [('mr', fit.matrix_lps), ('us', None)]:
                files[mod + '_image'] = f'{mod}/image.nii.gz'
                image = resample_volume(native[mod], reference, transform)
                write_nifti(image, stage / files[mod + '_image'], reference); del image
                output_masks[mod] = resample_volume(masks[mod], reference, transform, label=True)
                if not np.any(sitk.GetArrayViewFromImage(output_masks[mod])):
                    raise ValueError(f'{mod}: gland disappeared from the output grid')
                files[mod + '_prostate_label'] = f'{mod}/prostate_label.nii.gz'
                write_nifti(output_masks[mod], stage / files[mod + '_prostate_label'], reference)
                corners = mask_bounds(masks[mod])
                if transform is not None:
                    corners = apply_points(transform, corners)
                indices = (corners - np.asarray(reference.GetOrigin())) / np.asarray(reference.GetSpacing())
                if np.any(indices < -.5) or np.any(indices > np.asarray(reference.GetSize()) - .5):
                    warnings.append(f'{mod}: gland bounding box exceeds output grid; inspect possible clipping')
                combined = np.zeros(config.size[::-1], np.uint16)
                for number in sorted(targets[mod]):
                    moved = resample_volume(targets[mod][number], reference, transform, label=True)
                    name = f'{mod}/target_{number}_label.nii.gz'; files[f'{mod}_target_{number}'] = name
                    write_nifti(moved, stage / name, reference)
                    array = sitk.GetArrayViewFromImage(moved); count = int(array.sum())
                    if not count:
                        warnings.append(f'{mod} Target{number}: empty after nearest-neighbor resampling')
                    label_stats.append(dict(modality=mod, target_number=number, output_voxels=count))
                    combined[(array > 0) & (combined == 0)] = number
                if targets[mod]:
                    image = sitk.GetImageFromArray(combined); image.CopyInformation(reference)
                    name = f'{mod}/targets_label.nii.gz'; files[mod + '_targets_label'] = name
                    write_nifti(image, stage / name, reference)
            files['forward_transform'] = 'MR_to_US_LPS.tfm'
            files['inverse_transform'] = 'US_to_MR_LPS_for_resampling.tfm'
            sitk.WriteTransform(itk_transform(fit.matrix_lps), str(stage / files['forward_transform']))
            sitk.WriteTransform(itk_transform(np.linalg.inv(fit.matrix_lps)), str(stage / files['inverse_transform']))
            metrics = measure(masks, output_masks, targets, fit.matrix_lps, reference)
            if fit.diagnostics['rotation_warning']:
                warnings.append(f'Rotation exceeds {config.rotation_warning_degrees:g} degrees; retained for review')
            if metrics['target_tre_after_mm'] is not None and metrics['target_tre_after_mm'] > config.tre_warning_mm:
                warnings.append(f'Mean Target TRE exceeds {config.tre_warning_mm:g} mm; retained for review')
            verification = verify_outputs(stage, files, native, masks, reference, fit.matrix_lps)
            report = dict(status='success', version=__version__, case_id=case_id, input_signature=signature,
                reproducibility=identity, config=config.to_dict(), direction='MR_to_US', physical_units='mm',
                transform=dict(mr_to_us_lps_mm=fit.matrix_lps.tolist(), mr_to_us_ras_mm=fit.matrix_ras.tolist(),
                               us_to_mr_lps_mm=np.linalg.inv(fit.matrix_lps).tolist()),
                grid=dict(size=list(reference.GetSize()), spacing_mm=list(reference.GetSpacing()),
                          origin_lps_mm=list(reference.GetOrigin()), direction_lps=list(reference.GetDirection())),
                diagnostics=fit.diagnostics, metrics=metrics, labels=label_stats,
                warnings=warnings, requires_review=bool(warnings), verification=verification,
                files=files, output_sha256={name: file_hash(stage / name) for name in files.values()},
                completed_utc=datetime.now(timezone.utc).isoformat())
            (stage / 'registration.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            backup = None
            if output.exists():
                backup = output.with_name(f'.{output.name}-backup-{uuid.uuid4().hex}')
                output.rename(backup)
            try:
                stage.rename(output)
            except BaseException:
                if backup is not None:
                    backup.rename(output)
                raise
            if backup is not None:
                shutil.rmtree(backup)
            return RegistrationResult(output, report)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

