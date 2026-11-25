# Rigid initialization for prostate: MuReg

Mesh-based rigid registration of MuReg prostate MR-US pairs for subsequent
deformable registration. **MR is moving; US is fixed.** Registered MR images and
labels are resampled onto the preprocessed US grid.

Based on [dalbenzioG/Initial_Registration](https://github.com/dalbenzioG/Initial_Registration),
using its original mesh extraction, PCA-frame computation and rigid ICP routines,
with adapted PCA initialization and MuReg preprocessing. This is an adaptation,
not the unmodified upstream end-to-end pipeline.

## Install

Python 3.10 or newer. Runs on CPU; no GPU or MONAI installation is required.

```bash
git clone https://github.com/Msx00/rigid_initialization_monai.git
cd rigid_initialization_monai
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on Linux/macOS), then install:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

Exact versions from the tested MuReg environment are in
[requirements-tested.txt](requirements-tested.txt).

## Run

Each input split (`train`, `val` or `test`) contains matching filenames in
`mr_images`, `us_images`, `mr_labels` and `us_labels`. Images must be XYZ or XYZ1;
labels must be six-channel XYZ6 binary masks (0/1 or 0/255). Channel0 is the gland
used for fitting; channels1-5 are optional landmark masks.

```bash
mureg-rigid --data-root /path/to/mureg --out-dir /path/to/new_output --subsets train val test
mureg-rigid-verify --data-root /path/to/mureg --out-dir /path/to/new_output --subsets train val test
mureg-rigid-metrics --out-dir /path/to/new_output --subset test
```

Request only splits present in your input. Output must be new or empty.
Outputs use the preprocessed US grid, 128 x 128 x 128. Use `us_images` as fixed
and the already initialized `mr_images` as moving for the next registration
stage. **Do not apply the stored rigid transform again.** Forward matrices are
`T_mr2us_4x4`; inverse matrices are `T_us2mr_4x4`.

For two standalone 3D binary masks (transforms only):

```bash
mureg-rigid-masks --moving-mask mr_mask.nii.gz --fixed-mask us_mask.nii.gz --out-transform mr_to_us.json
```

## Parameters

| Option | Default |
| --- | ---: |
| `--pca-rotation-gain` | 0.10 |
| `--decimation-reduction` | 0.20 |
| `--smoothing` | 0 |
| `--landmarks` | 7000 |
| `--iterations` | 30 |
| `--max-rotation-degrees` | 30 |

The rotation threshold is an acceptance gate after fitting; it does not constrain
ICP. There is no Dice threshold. Rejected cases retain diagnostic transforms;
check acceptance counts before reporting results. The metrics command refuses
complete-cohort reporting when cases were rejected.

A JSON profile is available in [configs/best.json](configs/best.json), loaded with
`--registration-config configs/best.json`. Explicit flags override profile values.
The numeric defaults originated from historical US-to-MR train/validation
experiments and have not been reoptimized for MR-to-US. The filename does not
imply that this profile is optimal for the current direction. Earlier development
also used test data; see [method and provenance](docs/METHOD.md).

Distances use header-spacing-corrected world coordinates; millimetres have not
been established for the development data. Metric definitions and aggregation
rules are documented in the method guide.

## Documentation and checks

- [Full MuReg usage guide](docs/MUREG.md)
- [Data format and coordinates](docs/DATA_FORMAT.md)
- [Method, metrics and provenance](docs/METHOD.md)
- [MR-to-US direction migration](docs/MR_TO_US_MIGRATION.md)
- [Current release validation](docs/RELEASE_VALIDATION.md)

Run the synthetic functional tests after installation:

```bash
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
```

## Attribution and licence

Based on Initial_Registration commit
`21ee6ceb50bdb1e8d14072cb54ebb1bd2c611029`, under **Apache-2.0**. Original authors
retain their rights. Keep [LICENSE](LICENSE), [NOTICE](NOTICE) and
[UPSTREAM.json](UPSTREAM.json) when redistributing. Vendored upstream primitives
are unchanged. No patient images or case-level experiment files are included.
