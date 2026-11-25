> Run installation and CLI commands from the combined repository root.

# MONAI TCIA Rigid Alignment

Register prostate MRI to transrectal ultrasound in physical millimetres and export paired volumes for MONAI. Each case contains the resampled MR, fixed US, prostate masks, optional numbered lesion masks and both rigid transforms. A single command writes a relative-path `dataset.json` for the existing train, validation and test splits.

[Input CSV schema](tcia/DATA_FORMAT.md) · [Raw TCIA conversion](tcia/RAW_TCIA.md) · [Geometry](tcia/GEOMETRY.md)

## Try it with synthetic data

From the repository root, using Python 3.10+ in a virtual environment:

```bash
python -m pip install ".[tcia]"
tcia-rigid demo --output outputs/demo
tcia-rigid batch --monai --pairs outputs/demo/input/pairs.csv --output outputs/example
```

Open `outputs/example/dataset.json`. Its test split contains one synthetic case. The registration runs on CPU. For the library versions used in the reference experiment, use Python 3.11 and install with `python -m pip install -c requirements-tcia-tested.txt ".[tcia,dicom]"`.

## Export an existing split dataset

```bash
tcia-rigid batch --monai \
  --pairs registration_dataset/pairs.csv --data-root registration_dataset \
  --output outputs/MONAI_TCIA_Rigid_Alignment --workers 8
```

Supply the **native** moving MR and fixed US images and binary gland masks in NRRD or NIfTI. Within each modality, image and masks must share their grid. MR and US can have different native grids. Each row identifies one pair, its patient and a split (`train`, `val` or `test`). Use the same patient ID across all of that patient's pairs. The exporter rejects patients assigned to multiple splits. Relative input paths resolve against `--data-root`; optional matching Target IDs enable TRE evaluation. [Example manifest generator](../examples/tcia/write_manifest.py)

For original [TCIA Prostate-MRI-US-Biopsy data](https://www.cancerimagingarchive.net/collection/prostate-mri-us-biopsy/), install `python -m pip install '.[tcia,dicom]'`, then use `prepare` to convert the official pairings, DICOM and STL masks. If a split is needed:

```bash
tcia-rigid split --pairs data/prepared/pairs.csv --output data/splits --ratios .6 .1 .3 --seed 42
tcia-rigid batch --monai --pairs data/splits/pairs.csv --data-root data/prepared --output outputs/registered --workers 8
```

A new random split can have different patient membership from the reference experiment. Use an existing split manifest to preserve its assignments. Each worker uses one computation thread by default. Monitor `progress.json`; rerun the same command to resume verified case outputs when their settings, input hashes, environment and implementation match. Review any failures in `summary.json`. [Raw preparation commands](tcia/RAW_TCIA.md)

## Registration profile

| Parameter | TCIA default |
|---|---:|
| PCA rotation gain | 0 |
| Mesh smoothing iterations | 0 |
| Maximum ICP surface samples | 4000 |
| Maximum ICP iterations | 150 |
| Mesh decimation reduction | 20% |
| US-centred grid | 175 × 175 × 175 at 0.8 mm isotropic |

Initialization aligns the gland mesh centroids. Gain zero disables the initial PCA rotation; rigid ICP still estimates rotation and translation. The native gland surfaces drive the fit. Numbered Targets are reserved for evaluation. Images use linear interpolation and masks use nearest neighbour. The exported MR already includes the transform. The forward ITK transform maps native MR points into US in LPS millimetres. A rotation above 30° sets a review flag and the case is retained. CLI options override the [JSON profile](../configs/tcia.json). For one case, use `tcia-rigid pair --help` or the [Python API example](../examples/tcia/register_one.py).

## Output and loading

```text
outputs/registered/
  dataset.json                      # training / validation / test
  pairs.csv                         # successful case index
  progress.json, summary.json, export_check.json
  metrics/{metric_summary,pair_metrics,target_metrics}.csv
  <train|val|test>/<patient>/<pair_id>/
    mr/  us/                        # image.nii.gz, prostate_label.nii.gz
                                    # optional target_N_label.nii.gz and targets_label.nii.gz
    MR_to_US_LPS.tfm
    US_to_MR_LPS_for_resampling.tfm
    registration.json
```

Paths in `dataset.json` are relative to its parent. Each record uses explicit moving/fixed modality keys. To read the first training pair with MONAI installed in your downstream environment:

```python
import json
from pathlib import Path
from monai.transforms import LoadImaged

root = Path("outputs/registered")
case = json.loads((root / "dataset.json").read_text())["training"][0]
keys = ["mr_image", "us_image", "mr_prostate_label", "us_prostate_label"]
sample = LoadImaged(keys=keys, ensure_channel_first=True)(
    {k: str(root / case[k]) for k in keys}
)
```

For the synthetic example use `outputs/example` and the `test` key. The datalist is written after all requested cases pass export checks. Additional keys include per-modality Target paths (when present), transforms and a review flag.

## Reference results

With the selected settings, validation Target TRE was **7.32 mm** and mean gland Dice **0.8293**. Held-out test TRE was **7.47 mm** and Dice **0.8233**. TRE averages matching Target centroids per pair, then pairs per patient, then patients. Dice weights each pair equally. [Split counts, baseline and surface metrics](tcia/EXPERIMENTS.md)

## Tests and attribution

Run `python -m pip install '.[tcia,dicom]'
python -m unittest discover -s tests -v` after installation. For the optional DICOM tests install the dicom extra. The repository distributes code and synthetic examples. Obtain research data from TCIA under the collection's terms. Code is licensed under Apache-2.0; the vendored mesh/PCA/ICP routines and their upstream revision are credited in [NOTICE](../NOTICE) and [UPSTREAM.json](../UPSTREAM-TCIA.json). [License](../LICENSE)
