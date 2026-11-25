"""Reproducible synthetic data, never patient data."""
from pathlib import Path
import numpy as np
import SimpleITK as sitk


def make_demo(output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    z, y, x = np.indices((40, 42, 44))
    gland = (((x - 21) / 8) ** 2 + ((y - 20) / 10) ** 2 + ((z - 19) / 7) ** 2 < 1).astype(np.uint8)
    target = (((x - 24) / 2) ** 2 + ((y - 22) / 2) ** 2 + ((z - 18) / 2) ** 2 < 1).astype(np.uint8)
    ramp = (x + .4 * y + .2 * z + 30 * gland).astype(np.float32)
    shift = [8., -6., 4.]
    for mod, origin in [('mr', [-20., -18., -15.]), ('us', [-12., -24., -11.])]:
        folder = output / mod; folder.mkdir(exist_ok=True)
        for name, array in [('image', ramp), ('prostate_label', gland), ('target_1_label', target)]:
            image = sitk.GetImageFromArray(array); image.SetSpacing([1., 1.2, 1.4]); image.SetOrigin(origin)
            sitk.WriteImage(image, str(folder / f'{name}.nrrd'), True)
    import csv, json
    row = dict(pair_id='synthetic-001', patient='synthetic', split='test', mr_image='mr/image.nrrd', us_image='us/image.nrrd',
        mr_prostate_label='mr/prostate_label.nrrd', us_prostate_label='us/prostate_label.nrrd',
        mr_targets=json.dumps({'1': 'mr/target_1_label.nrrd'}), us_targets=json.dumps({'1': 'us/target_1_label.nrrd'}))
    with (output / 'pairs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    matrix = np.eye(4); matrix[:3, 3] = shift
    (output / 'ground_truth.json').write_text(json.dumps(dict(mr_to_us_lps_mm=matrix.tolist(), synthetic=True), indent=2))
    return row

