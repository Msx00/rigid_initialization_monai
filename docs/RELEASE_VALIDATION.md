# MuReg release 0.4.0 validation

This release contains the MuReg MR-to-US workflow and its four commands:
`mureg-rigid`, `mureg-rigid-masks`, `mureg-rigid-verify` and `mureg-rigid-metrics`.
The fitting, preprocessing, resampling, verification and metric implementations
are unchanged from version 0.3.0. The package exposes only `mureg_rigid` and
`pca_icp`, with NumPy, SciPy, NiBabel, VTK and scikit-image as dependencies.

A fresh wheel was built and installed into an isolated target on Windows with
Python 3.12. All **5 synthetic functional tests passed**:

1. Known world-coordinate translation with zero-gain initialization.
2. Binary label encodings and rejection of unsupported formats.
3. The standalone mask command and transformation diagnostics.
4. MR-to-US batch registration, output verification and metric evaluation,
   including rejection of historical direction metadata.
5. A known 20-degree MR-to-US rotation with anisotropic spacing, fixed US output
   grid, forward/inverse matrices and resampled labels.

All four installed commands passed their help checks. The wheel contains only
`mureg_rigid` and `pca_icp`; its entry points and five dependencies match the
MuReg-only configuration. No additional workflow packages or optional dependency
extras are included. Documentation links and the English-only source check passed.
The original upstream file hashes match `UPSTREAM.json`.

These are packaging and regression checks; no new clinical registration experiment
was run for this cleanup. Previously generated datasets and the MR-to-US test
results are unaffected. Hosted Linux/Windows CI is configured to install the
base package and run the same five tests after pushing.
