# Rigid Initialization for MONAI: MuReg and TCIA

One repository for two prostate **MR-to-US** rigid registration workflows.
**MR is moving; US is fixed in both workflows.** Each workflow
keeps its own input format, geometry, parameters and evaluation definitions.
Both use mesh-based registration built on
[dalbenzioG/Initial_Registration](https://github.com/dalbenzioG/Initial_Registration),
with adapted PCA initialization and the original rigid ICP routines.

## Choose a Workflow

| | MuReg | TCIA Prostate-MRI-US-Biopsy |
| --- | --- | --- |
| Command | `mureg-rigid` | `tcia-rigid` |
| Moving -> fixed | **MR -> US** | **MR -> US** |
| Inputs | Matched NIfTI images and six-channel labels | CSV pairs manifest; native MR/US images and binary gland masks; optional Target masks |
| Output grid | Preprocessed US grid, 128 x 128 x 128 | US-gland-centred, 175 x 175 x 175 at 0.8 mm isotropic |
| Default PCA gain | 0.10 | 0 (centroid initialization) |
| Default ICP sampling / iterations | 7000 / 30 | 4000 / 150 |
| Smoothing / mesh reduction | 0 / 20% | 0 / 20% |
| Rotation above 30 degrees | Case rejected | Case retained with a review flag |
| Distances | Corrected affine world units; mm not established for development data | Physical millimetres with validated image geometry |
| Guide | [MuReg](docs/MUREG.md) | [TCIA](docs/TCIA.md) / [Chinese](docs/TCIA.zh-CN.md) |

Do not swap these workflows' transforms or preprocessing assumptions. MONAI
itself is not required to produce the outputs; install it in your downstream
training environment if needed.

## Installation

Python 3.10 or newer. CPU execution; no GPU required.

```bash
git clone https://github.com/Msx00/rigid_initialization_monai.git
cd rigid_initialization_monai
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in Windows PowerShell,
`source .venv/bin/activate` on Linux/macOS), then install:

```bash
python -m pip install --upgrade pip
# MuReg only:
python -m pip install .
# Both MuReg and prepared TCIA data:
python -m pip install ".[tcia]"
# Also enable original TCIA DICOM/STL preparation:
python -m pip install ".[tcia,dicom]"
```

Use `-e` for editable development installs. The two original tested environments
are recorded separately in [requirements-tested.txt](requirements-tested.txt)
and [requirements-tcia-tested.txt](requirements-tcia-tested.txt); do not combine
their exact version pins into one constraints file.

## Quick Start: MuReg

Input splits contain matching filenames in `mr_images`, `us_images`, `mr_labels`
and `us_labels`. Images are XYZ or XYZ1; binary labels are XYZ6 (0/1 or 0/255).
Channel0 is the gland used for fitting; channels1-5 are optional landmark masks.

```bash
mureg-rigid --data-root /path/to/mureg --out-dir /path/to/new_output --subsets train val test
mureg-rigid-verify --data-root /path/to/mureg --out-dir /path/to/new_output --subsets train val test
mureg-rigid-metrics --out-dir /path/to/new_output --subset test
```

Request only splits present in your input. Output must be new or empty. Use
`us_images` as fixed and the already initialized `mr_images` as moving for the
next registration stage. A JSON fitting profile is in [configs/best.json](configs/best.json). Its numeric
defaults were selected for the historical US-to-MR direction and have not been
reoptimized for MR-to-US; previous accuracy results do not validate this direction.
For a pair of standalone binary masks:

```bash
mureg-rigid-masks --moving-mask mr_mask.nii.gz --fixed-mask us_mask.nii.gz --out-transform mr_to_us.json
```

Full options, input/output layout, coordinate conventions and provenance:
[MuReg guide](docs/MUREG.md).

## Quick Start: TCIA

First try the self-contained synthetic example:

```bash
tcia-rigid demo --output outputs/tcia_demo
tcia-rigid batch --monai --pairs outputs/tcia_demo/input/pairs.csv --output outputs/tcia_example
```

For converted native data with an existing patient-disjoint split manifest:

```bash
tcia-rigid batch --monai --pairs /path/to/pairs.csv --data-root /path/to/prepared_data --output outputs/tcia_registered --workers 8
```

The exporter writes a relative-path `dataset.json` with `training`, `validation`
and `test`. Records contain explicit `mr_image`, `us_image`, gland/Target label
and transform paths. MR has already been transformed to the fixed US grid;
**do not apply its rigid transform again**. The exporter checks geometry,
patient split separation and case output integrity, supports verified resume,
and reports failures in `summary.json`.

Use `tcia-rigid pair --help` for one pair, `prepare --help` for original TCIA
DICOM/STL conversion, and `split --help` to generate patient-disjoint splits.
Original raw conversion requires the `dicom` extra. Preserve an existing split
manifest when reproducing an experiment instead of creating a new random split.

See the [TCIA guide](docs/TCIA.md), [CSV schema](docs/tcia/DATA_FORMAT.md),
[raw conversion](docs/tcia/RAW_TCIA.md), [physical geometry](docs/tcia/GEOMETRY.md)
and [reference experiments](docs/tcia/EXPERIMENTS.md).

## Commands and Configuration

Version 0.3.0 changes MuReg fitting and resampling from US-to-MR to MR-to-US.
CLI names and input layouts are retained, but outputs and matrix roles change.
Regenerate outputs in a new directory; do not relabel earlier datasets.
See [direction migration](docs/MR_TO_US_MIGRATION.md). `tcia-rigid` adds the `pair`, `batch`,
`convert`, `prepare`, `split` and `demo` subcommands. All support `--help`.
CLI flags override each workflow's own JSON configuration:

- MuReg: `--registration-config configs/best.json`; final gate is a separate
  `--max-rotation-degrees` option.
- TCIA: `--config configs/tcia.json`; use `--rotation-warning-degrees` for its
  review flag. Do not pass a MuReg profile to TCIA or vice versa.

The workflows' metrics also differ. MuReg pools saved label-centroid distances;
TCIA Target TRE averages Targets within pairs, pairs within patients, then
patients. Do not compare those reported TRE values as if they used one definition.
The MuReg development history and test-data-use disclosure remain documented in
[its method guide](docs/METHOD.md).

## Development

Install the complete test dependencies and run both synthetic suites:

```bash
python -m pip install ".[tcia,dicom]"
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
```

The supplied TCIA registration modules retain their original implementation in
`src/tcia_prostate_rigid/`. One Windows-compatible DICOM path comparison was
adjusted; [TCIA-INTEGRATION.json](TCIA-INTEGRATION.json) records original and
integrated source hashes and this platform-only change. Its private vendor namespace avoids collisions with MuReg's
`pca_icp` package. No patient images or case-level experiment files are included. See [combined validation](docs/INTEGRATION_VALIDATION.md).

## Attribution and License

Based on Initial_Registration commit
`21ee6ceb50bdb1e8d14072cb54ebb1bd2c611029`, under **Apache-2.0**. Original authors
retain their rights. This package modifies initialization/orchestration; it is
not the unmodified upstream end-to-end pipeline. See [NOTICE](NOTICE),
[LICENSE](LICENSE), [MuReg upstream hashes](UPSTREAM.json),
[TCIA upstream hashes](UPSTREAM-TCIA.json) and the per-workflow guides.
Obtain TCIA research data under the collection's own terms; the code license
does not grant rights to redistribute those data.

Release checks: [MR-to-US 0.3.0](docs/MR_TO_US_MIGRATION.md) and
[historical combined 0.2.0 validation](docs/INTEGRATION_VALIDATION.md).
