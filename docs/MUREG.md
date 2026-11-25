# MuReg PCA + ICP Rigid Registration

Rigid initialization of prostate MR–US pairs for subsequent deformable registration.
**US is moving; MR is fixed.**

This project adapts the PCA + ICP workflow from
[dalbenzioG/Initial_Registration](https://github.com/dalbenzioG/Initial_Registration),
using its original mesh-processing, PCA-frame computation and rigid-ICP functions.
The adaptation adds MuReg data handling, modified PCA initialization and rotation
damping. The differences are described below and in [METHOD.md](METHOD.md).

## Choose a Mode

| Your input | Command | Result |
| --- | --- | --- |
| MR/US images and six-channel MuReg labels | `mureg-rigid` | Aligned NIfTI images, labels and per-case transforms |
| A pair of 3D binary masks | `mureg-rigid-masks` | Forward/inverse transforms and diagnostics in JSON |
| Existing batch outputs | `mureg-rigid-verify` | File, grid, transformation and label-preservation checks |
| Complete batch outputs | `mureg-rigid-metrics` | Per-case and aggregate TRE, Chamfer, Dice and HD95 |

## Prerequisites and Installation

Python **3.10+**, with NumPy, SciPy, NiBabel, VTK and scikit-image. Dependencies
are installed automatically. Execution is CPU-based; no GPU is required.

Download or clone **this repository**, then run from its root:

```bash
python -m pip install .
```

A dedicated environment is recommended. For example, in Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

On Linux/macOS, activate with `source .venv/bin/activate` instead. Use
`python -m pip install -e .` for development. The local release was tested with
Python 3.12 on Windows; exact dependency versions are in [requirements-tested.txt](../requirements-tested.txt).

## Quick Start

### 1. Batch registration

```bash
mureg-rigid --data-root /path/to/dataset --out-dir /path/to/rigid_output --subsets test
```

Windows example:

```powershell
mureg-rigid --data-root "D:\datasets\mureg" --out-dir "D:\results\rigid_output" --subsets test
```

Use `--subsets train val test` to process all three splits, or request only the
splits present in your dataset. The output directory must be **new or empty**.
`python -m mureg_rigid` is equivalent to `mureg-rigid`.

### 2. Verify outputs and calculate metrics

```bash
mureg-rigid-verify --data-root /path/to/dataset --out-dir /path/to/rigid_output --subsets test
mureg-rigid-metrics --out-dir /path/to/rigid_output --subset test
```

Verification checks accepted outputs and lists rejected cases. Check the
accepted/total counts before reporting complete-cohort performance. The metrics
command refuses a split containing rejected cases.

### 3. Register a pair of binary masks

```bash
mureg-rigid-masks --moving-mask us_mask.nii.gz --fixed-mask mr_mask.nii.gz --out-transform us_to_mr.json
```

Masks must be nonempty **3D binary NIfTI files**, encoded as 0/1 or 0/255. This
mode honours the supplied affines directly, without the batch mode's MuReg
spacing correction, 128³ crop or intensity normalization. It writes transforms
and diagnostics, not aligned NIfTI volumes. Choose a new output JSON file.

## Data Layout Expectations

```text
dataset/
├── train/
│   ├── mr_images/case001.nii.gz
│   ├── us_images/case001.nii.gz
│   ├── mr_labels/case001.nii.gz
│   └── us_labels/case001.nii.gz
├── val/                       # same four subfolders, if requested
└── test/                      # same four subfolders, if requested
```

- Corresponding files must have **identical names** in all four folders.
- Images: `XYZ` or `XYZ1` NIfTI arrays.
- Labels: **six-channel `XYZ6` binary masks**, encoded as 0/1 or 0/255.
- Channel0 is the prostate mask used for fitting; channels1–5 are optional landmark masks.
- Integer class maps, channel-first arrays and probability maps require conversion.
- Each image must share its shape and affine with its corresponding labels.
  MR and US may have different grids.

Batch preprocessing reorients to RAS, applies the historical MuReg spacing
correction, centre-pads/crops each volume to 128³ and normalizes image intensity
to [0,1]. The crop is around the **volume centre**, not the prostate centroid.
This is a dataset-specific preprocessing path. See [DATA_FORMAT.md](DATA_FORMAT.md).

## Main Options

Defaults shared by both registration commands:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--pca-rotation-gain` | **0.10** | Fraction of the initial PCA rotation vector to apply, from 0 to 1 |
| `--decimation-reduction` | **0.20** | Target fraction of the mesh to remove, in [0,1) |
| `--smoothing` | **0** | Mesh smoothing iterations |
| `--landmarks` | **7000** | Maximum mesh points sampled by ICP; not anatomical landmark labels |
| `--iterations` | **30** | Maximum ICP iterations; convergence can occur earlier |
| `--max-rotation-degrees` | **30** | Acceptance threshold on the final rigid rotation |

There is **no Dice threshold**. The angle gate is applied **after** registration;
it does not constrain the optimizer. Gain0 removes the PCA initial rotation but
retains centroid alignment; ICP can still estimate rotation afterwards.

Override settings directly:

```bash
mureg-rigid --data-root /path/to/dataset --out-dir /path/to/experiment --subsets test --pca-rotation-gain 0.5 --smoothing 10 --landmarks 5000 --iterations 60 --max-rotation-degrees 45
```

Or load the included five-field fitting profile with the batch command:

```bash
mureg-rigid --data-root /path/to/dataset --out-dir /path/to/profile_output --subsets test --registration-config configs/best.json --max-rotation-degrees 30
```

Explicit flags override JSON; JSON overrides defaults. The angle gate is a
separate flag, not a profile JSON field. `--selection-dataset` records a profile's
parameter-selection origin, not the input split; its default is `train_val`.
Use `--selection-dataset unspecified` for a custom profile of unknown origin.
Run any command with `--help` for all options.

## Outputs by Mode

### Batch NIfTI registration

Each requested split contains:

| Folder | Contents |
| --- | --- |
| `mr_images/` | Fixed MR images |
| `us_images/` | US images already rigidly resampled to the MR grid |
| `mr_labels/`, `us_labels/` | Fixed and initialized six-channel labels |
| `mr_labels_maps/`, `us_labels_maps/` | Derived argmax label maps |
| `us_images_before_rigid/` | Preprocessed US images before the rigid transform |
| `registration_masks/` | Exact binary channel0 masks used for fitting |
| `meta/` | Per-case transforms, PCA axes, diagnostics and acceptance summaries |

The output root also contains `run_config.json`, recording settings and provenance.
Images are float32 `128×128×128×1`; six-channel labels are uint8
`128×128×128×6`, foreground255. Derived argmax maps collapse overlaps and cannot
distinguish channel0 from background; retain six-channel masks for evaluation.

Rejected or failed cases retain `*_failed.json` metadata. Exit code **2** means
at least one case was rejected or failed; inspect `failure_reason` and any
`exception` in `meta/`.

### Binary-mask registration

The specified JSON contains `T_moving_to_fixed`, `T_fixed_to_moving`, acceptance,
settings and diagnostics. A rejected fit returns exit code2 and retains its
diagnostic transform. This mode does not export resampled images.

## Transform Conventions

Transforms use homogeneous **column-vector world coordinates**:

```text
p_MR = T_us2mr @ p_US
T_us2mr = T_icp @ T_initial
```

Batch metadata stores `T_us2mr_4x4` and `T_mr2us_4x4`. The mask command uses
`T_moving_to_fixed` and `T_fixed_to_moving`. These are not voxel-index matrices.
Resampling uses the inverse map from the fixed grid into the moving image;
images use linear interpolation, labels use nearest-neighbour.

For subsequent deformable registration, use `mr_images` as fixed and `us_images`
as the initialized moving input. **Do not apply the stored rigid matrix again**
to `us_images` or `us_labels`.

## Changes from the Original Workflow

| Component | Original PCA+ICP workflow | This adaptation |
| --- | --- | --- |
| PCA-axis correspondence | Fixed axis order; four proper sign candidates | Six axis permutations and sign choices; 24 proper rotations |
| Candidate selection | Symmetric surface-distance score | Smallest rotation angle |
| Initial rotation | Full selected PCA rotation | Rotation-vector scaling by the configured gain |
| ICP starts | Conditional multistart for unstable PCA | One selected initialization |
| Acceptance | Original workflow's reporting | Final rotation gate; no Dice gate |

The original mesh-processing, PCA-frame and ICP source files are vendored without
modification in `src/pca_icp/`. Adaptations are in `src/mureg_rigid/`.
**Gain1 does not reproduce the original end-to-end pipeline**, because candidate
generation and selection differ. See [METHOD.md](METHOD.md) and the file
hashes in [UPSTREAM.json](../UPSTREAM.json).

## Evaluation and Default-Profile Provenance

The metrics command writes `<split>_four_metrics.json`, including per-case
`channel_errors` and aggregate measurements:

- **TRE:** corresponding nonempty saved-label centroid distances, pooled across
  channels0–5 and cases. For labels1–5, pool `channel_errors` excluding key `0`.
  Label0 also participates in fitting.
- **Chamfer and HD95:** bidirectional nearest-neighbour distances between
  channel0 surfaces. Full definitions are in [METHOD.md](METHOD.md).
- **Dice:** channel0 mask overlap.
- TRE is pooled by valid channel pair; surface metrics are summarized across
  cases. Population SD uses `ddof=0`.

The verifier also reports native labels1–5 TRE averaged within cases and then
across cases; this differs from pooled saved-label TRE. Distance units are affine
world coordinates. **Millimetres require confirmed spatial units and calibration.**

The default profile was selected on **65 training + 8 validation cases** using
pooled labels1–5 TRE, requiring all cases and valid paired labels to be retained.
Earlier development also used test data. These results are not an untouched
independent-test evaluation. Validate parameter choices on your own data before
final testing. See [method provenance](METHOD.md).

## Troubleshooting

- **Command not found:** activate the installation environment; for batch mode,
  try `python -m mureg_rigid --help`.
- **Missing/unpaired files:** check split names and matching `.nii.gz` filenames.
- **Unexpected label shape:** batch labels must be `XYZ6`; use the mask command
  for standalone 3D binary masks.
- **Empty or lost labels:** inspect the volume-centred crop and run verification.
- **Rotation threshold not met:** inspect the transform and acceptance limit;
  changing the gate filters results rather than changing ICP optimization.
- **VTK import failure:** check your Python environment and local library policy.
- **Source-file import errors:** install the project and use the documented
  commands instead of executing individual source files by path.

## Development and Detailed Documentation

```bash
python -m pip install '.[tcia,dicom]'
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
```

- [Method and metric definitions](METHOD.md)
- [Data and coordinate conventions](DATA_FORMAT.md)
- [Release validation](VALIDATION.md)
- [GitHub publishing instructions](PUBLISHING.md)

The source release includes code, configuration, documentation and synthetic tests.
Clinical NIfTI data and case-level experiment outputs are not included.

## Acknowledgements and License

This work builds on **Initial_Registration** by the original repository's authors:

- Repository: https://github.com/dalbenzioG/Initial_Registration
- Upstream base commit: `21ee6ceb50bdb1e8d14072cb54ebb1bd2c611029`
- License: **Apache-2.0**; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE).

Please acknowledge the original repository and describe the adapted initialization
when using this package in research. Retain the license, notice and attribution
when redistributing the code.
