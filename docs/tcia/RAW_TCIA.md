# Preparing original TCIA DICOM and STL

Install the optional reader:

```bash
python -m pip install '.[dicom]'
```

## Whole downloaded collection

```bash
tcia-rigid prepare \
  --metadata downloaded/Prostate-MRI-US-Biopsy-v2/metadata.csv \
  --targets downloaded/Supplementary_Data/Target-Data_2019-12-05-2.xlsx \
  --biopsy downloaded/Supplementary_Data/TCIA-Biopsy-Data_2020-07-14.xlsx \
  --stls downloaded/STL_Files_Segmentations/STLs.zip \
  --output data/prepared
```

`--biopsy` is optional. Use `--stls` with the ZIP itself or a directory containing extracted `.STL` files. Downloaded `File Location` paths are resolved relative to metadata.csv's parent, or `--dicom-root` if specified. ZIP members are read without extracting arbitrary paths.

The official workbooks supply MR/US SeriesInstanceUID pairs. Metadata and STL patient/UIDs are cross-checked. Preparation converts every required series once, writes native-grid NRRDs under `series/<patient>/<modality>/<uid>/`, and produces a portable `pairs.csv`. It defaults to T2 descriptions; `--mr-types T1 T2` broadens the filter. ADC/DWI/diffusion descriptions are excluded. Filtering is based on series descriptions and should be reviewed for unusual acquisitions.

Preparation does not register or normalize images. Original files stay unchanged. Existing identical conversions are reused; incompatible conversion folders are rejected. Missing series/meshes, incomplete DICOM file counts, or unreadable geometry are recorded in `prepare_report.json`, along with filtered sequences. A nonzero error result must be reviewed; it is not an assertion that every downloaded pair is usable.

```bash
tcia-rigid batch --pairs data/prepared/pairs.csv --data-root data/prepared \
  --output outputs/registered --workers 8
```

## One series

```bash
tcia-rigid convert --modality mr --dicom-dir downloaded/mr-series \
  --prostate-stl downloaded/mr-prostate.STL \
  --target-stl 1=downloaded/mr-target1.STL --output data/mr

tcia-rigid convert --modality us --dicom-dir downloaded/us-series \
  --prostate-stl downloaded/us-prostate.STL \
  --target-stl 1=downloaded/us-target1.STL --output data/us
```

Use `--series-uid` when a directory has multiple series. MR uses GDCM physical-position ordering, not filenames or InstanceNumber. Clearly missing planes on a consistent regular lattice are linearly interpolated and explicitly recorded; ambiguous irregular geometry is rejected. Some compressed DICOM transfer syntaxes may need an additional compatible pydicom decoder.

The US reader is specifically for **Eigen Artemis**, not arbitrary US vendors. Do not use ordinary DICOM volume spacing/orientation guesses for these multiframe objects. STL assumes explicit SPACE metadata or default LPS; the US voxel axis permutation is a separate step. See [geometry details](GEOMETRY.md).

Python conversion helpers:

```python
from tcia_prostate_rigid.raw import convert_series, read_dicom_series, stl_to_mask
image, info = read_dicom_series("downloaded/us-series", "us")
mask, diagnostics = stl_to_mask("downloaded/us-prostate.STL", image, modality="us")
```

## Patient-disjoint splits

```bash
tcia-rigid split --pairs data/prepared/pairs.csv --output data/splits \
  --ratios .6 .1 .3 --seed 42
tcia-rigid batch --pairs data/splits/pairs.csv --data-root data/prepared \
  --output outputs/registered --workers 8
```

The split shuffles whole patients, not individual pairs, and writes new manifests without copying images. Pair counts need not match ratios exactly. This simple splitter does not reproduce a different study's pair-count balancing swaps or undistributed split files. Reuse your established manifests if exact cohorts matter. Select registration settings using training/validation data and keep the test split untouched during selection.

