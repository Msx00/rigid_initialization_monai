"""Optional DICOM/STL conversion; imports pydicom only when used."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import numpy as np
import SimpleITK as sitk
from filelock import FileLock
from .geometry import affine_lps
from .pipeline import file_hash


def read_dicom_series(directory, modality, series_uid=None):
    from ._conversion import pydicom, read_mr, read_us
    if pydicom is None:
        raise ImportError("DICOM conversion requires pydicom>=3.0; install it or run pip install '.[dicom]' from the repository root")
    directory = Path(directory).resolve()
    files = sorted(str(p) for p in directory.iterdir() if p.suffix.lower() == '.dcm')
    if not files:
        raise ValueError(f'No .dcm files in series directory: {directory}')
    headers = [(p, pydicom.dcmread(p, stop_before_pixels=True)) for p in files]
    uids = {str(h.SeriesInstanceUID) for _, h in headers}
    if series_uid is None:
        if len(uids) != 1:
            raise ValueError('Directory contains multiple series; provide series_uid')
        series_uid = next(iter(uids))
    selected = [(p, h) for p, h in headers if str(h.SeriesInstanceUID) == series_uid]
    if not selected:
        raise ValueError('Requested SeriesInstanceUID was not found')
    patients = {str(h.PatientID) for _, h in selected}
    if len(patients) != 1:
        raise ValueError('Mixed patients in DICOM series')
    modality = modality.lower()
    if modality not in ['mr', 'us']:
        raise ValueError('modality must be MR or US')
    if any(str(h.Modality).lower() != modality for _, h in selected):
        raise ValueError('Requested modality does not match the DICOM objects')
    series = dict(files=[p for p, _ in selected], uid=series_uid, patient=next(iter(patients)), source_dir=str(directory))
    image, info = (read_us(series) if modality == 'us' else read_mr(series))
    return image, dict(**info, series_uid=series_uid, patient=series['patient'], modality=modality)


def stl_to_mask(stl, reference, *, modality='mr'):
    """Binary TCIA STL -> label, explicit SPACE=RAS or default LPS."""
    from ._conversion import rasterize
    raw = stl if isinstance(stl, bytes) else Path(stl).read_bytes()
    return rasterize(raw, reference, modality.lower())


def convert_series(dicom_dir, prostate_stl, output_dir, *, modality, target_stls=None, series_uid=None):
    """Convert one TCIA series and matched STL surfaces into native-grid NRRDs."""
    output = Path(output_dir).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output == Path(dicom_dir).resolve() or output in Path(dicom_dir).resolve().parents:
        raise ValueError('Conversion output must not contain the original DICOM directory')
    raw = {'prostate': prostate_stl if isinstance(prostate_stl, bytes) else Path(prostate_stl).read_bytes()}
    targets = dict(target_stls or {})
    if any(type(n) is not int or not 1 <= n <= 65535 for n in targets):
        raise ValueError('Target IDs must be integers from 1 to 65535')
    for n, value in targets.items():
        raw[f'target_{n}'] = value if isinstance(value, bytes) else Path(value).read_bytes()
    source = {str(p.name): file_hash(p) for p in sorted(Path(dicom_dir).glob('*')) if p.suffix.lower() == '.dcm'}
    signature = hashlib.sha256(json.dumps(dict(files=source, stls={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
        reader_sha256=file_hash(Path(__file__).with_name('_conversion.py')), wrapper_sha256=file_hash(__file__),
        modality=modality.lower(), uid=series_uid, convention='explicit_SPACE_else_LPS_v1'), sort_keys=True).encode()).hexdigest()
    with FileLock(str(output.parent / f'.{output.name}.lock'), timeout=0):
        done = output / 'conversion.json'
        if done.exists():
            old = json.loads(done.read_text())
            if old['input_signature'] != signature:
                raise FileExistsError('Existing conversion has different inputs; use another output directory')
            if all((output / name).is_file() for name in old['files']):
                return old
            raise FileExistsError('Existing conversion is incomplete; use a fresh output directory')
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Conversion output is nonempty')
        image, info = read_dicom_series(dicom_dir, modality, series_uid)
        stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
        try:
            sitk.WriteImage(image, str(stage / 'image.nrrd'), True)
            files = ['image.nrrd']; labels = []; warnings = list(info.get('geometry_warnings', []))
            if info.get('spacing_conflict'):
                warnings.append('Eigen standard/private spacing conflict; standard PixelAspectRatio/1000 used')
            combined = np.zeros(image.GetSize()[::-1], np.uint16)
            for key, data in sorted(raw.items(), key=lambda item: -1 if item[0] == 'prostate' else int(item[0].split('_')[1])):
                mask, qa = stl_to_mask(data, image, modality=modality)
                filename = 'prostate_label.nrrd' if key == 'prostate' else key + '_label.nrrd'
                sitk.WriteImage(mask, str(stage / filename), True); files.append(filename)
                labels.append(dict(name=key, filename=filename, **qa))
                if qa['fraction_mesh_vertices_outside_image']:
                    warnings.append(f'{key}: mesh extends outside original image; label clipped')
                if qa['boundary_or_nonmanifold_edges']:
                    warnings.append(f'{key}: open/nonmanifold mesh')
                if qa.get('nonfinite_stl_normals_recomputed'):
                    warnings.append(f'{key}: invalid stored normals recomputed; vertices unchanged')
                if not .7 < qa['voxel_to_mesh_volume_ratio'] < 1.3:
                    warnings.append(f'{key}: rasterized/mesh volume ratio outside 0.7..1.3')
                if key.startswith('target_'):
                    a = sitk.GetArrayViewFromImage(mask)
                    combined[(a > 0) & (combined == 0)] = int(key.split('_')[1])
            if targets:
                label = sitk.GetImageFromArray(combined); label.CopyInformation(image)
                sitk.WriteImage(label, str(stage / 'targets_label.nrrd'), True); files.append('targets_label.nrrd')
            report = dict(status='success', input_signature=signature, info=info, warnings=warnings,
                          files=files, labels=labels, native_voxel_to_lps_mm=affine_lps(image).tolist())
            (stage / 'conversion.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if output.exists():
                output.rmdir()
            stage.rename(output)
            return report
        finally:
            if stage.exists():
                shutil.rmtree(stage)

