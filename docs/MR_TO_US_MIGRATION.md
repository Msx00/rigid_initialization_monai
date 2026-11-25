# MR-to-US direction change (0.3.0)

MuReg uses **MR moving -> US fixed**. It fits the channel0 MR mask directly
to the channel0 US mask through the same mesh extraction, adapted PCA and rigid
ICP calls. It does not obtain its fit by inverting a previous US-to-MR result;
directional ICP fits need not be inverses of each other.

## Output and API changes

- MuReg fixed images/labels: `us_images`, `us_labels` on the preprocessed US grid.
- MuReg warped images/labels: `mr_images`, `mr_labels`, resampled onto that grid.
- Before-registration debug images: `mr_images_before_rigid`.
- Forward matrix: `T_mr2us_4x4`; inverse matrix: `T_us2mr_4x4`.
- Case and run metadata explicitly record `MR_to_US`; output reference is US.
- PCA diagnostic frame keys are modality-independent `moving` and `fixed`.
- Verification and metrics reject metadata from earlier MuReg directions to
  prevent mixing historical and new results. Use the matching old release for
  old outputs. Start a fresh output directory for new registration runs.
- Downstream registration should use US as fixed and the already resampled MR
  as moving. Do not apply the rigid transform a second time.

## Parameter provenance

The MuReg numeric defaults (gain0.10, reduction0.20, smoothing0, 7000 points,
30 ICP iterations, 30-degree acceptance gate) originated from US-to-MR train/val
experiments. They remain available as a starting profile; no MR-to-US parameter
search has been run in this release. The historical `configs/best.json` filename
does not imply these parameters are optimal for the new direction. The default
current-direction `selection-dataset` is `unspecified`; run metadata separately
records the historical profile origin. Old ablation curves, TRE and surface
metrics must not be relabelled as MR-to-US results.

All previous clinical datasets and outputs remain separate and unchanged.
The packaged source contains no clinical data.

## Validation

The 0.3.0 wheel was built and installed into an isolated workspace target on
Windows/Python 3.12. All **5 MuReg synthetic tests passed**. They cover MR-to-US
translation, a known 20-degree rotation with anisotropic voxel spacings, the
fixed US grid, resampled labels, forward/inverse matrices, verification/metrics
and rejection of historical direction metadata. All four MuReg entry points
passed their help checks. Vendored upstream primitive hashes remained unchanged.
See [current release checks](RELEASE_VALIDATION.md).

A two-case real-data smoke check ran the default MR-to-US profile. One case
passed (rotation 4.571 degrees); one was rejected at the unchanged 30-degree gate
(rotation 38.767 degrees). Direct MR-mask-to-US-mask fits matched both stored
forward matrices. The accepted case passed file/grid/label verification, and
the metrics command correctly refused complete-cohort reporting with a rejected
case. This limited check is not a new clinical accuracy study, and the gate was
not relaxed to make the cases pass. No full-cohort rerun or parameter search
was performed.
