# Method, evaluation and provenance

## Registration

The vendored `pca_icp` files come from Initial_Registration commit
`21ee6ceb50bdb1e8d14072cb54ebb1bd2c611029`, under Apache-2.0. Their source files
are unmodified; hashes are recorded in `UPSTREAM.json`. Cite and acknowledge
[the original repository](https://github.com/dalbenzioG/Initial_Registration).

Batch fitting extracts channel0, calls the author's marching-cubes mesh loader,
optional windowed-sinc smoothing and topology-preserving decimation, and uses
the original `compute_pca_frame` to obtain mesh-centroid PCA frames. Each frame's
columns are its principal axes, sorted by descending eigenvalue and corrected
to a right-handed basis.

The adaptation enumerates six axis permutations and eight direction-sign
combinations. Removing determinant-negative reflections leaves 24 proper
rotations. The candidate with the smallest rotation angle is selected, then
its rotation vector is multiplied by the configured gain. Translation is
recomputed as `centroid_fixed - R @ centroid_moving`; translation is not damped.
The author's `run_icp` refines this one initialization in rigid mode. The
maximum mean update distance is 0.001; the configured iteration count is an
upper bound, not a guarantee that every run executes that many iterations.

This differs from the original complete pipeline, which holds principal-axis
order fixed, scores four proper sign candidates by surface distance, and can
run conditional multistart ICP. The adapter does not call that original
end-to-end pipeline. A gain of 1 therefore does not reproduce vanilla upstream
PCA initialization. Gain0 is a centroid-only initialization ablation, followed
by the same rigid ICP; it does not forbid subsequent rotation.

Only the two channel0 masks enter each fit. Dice and landmark coordinates do
not select that case's transform. The final rotation angle is used for
acceptance after optimization; Dice is evaluation-only. No per-case fallback
or alternate parameter search is performed by the published batch command.

Per-case metadata includes selected axis order/signs, the original PCA angle,
PCA centroids/eigenvalues/eigenvectors, the damped initial matrix, the ICP matrix,
forward/inverse rigid transforms and acceptance. A matrix with determinant -1
is a reflection, not a valid rigid rotation; negative individual axis components
alone do not make a transformation invalid.

## Default profile and development

The default gain0.10/smoothing0/points7000/iterations30/reduction0.20 profile
and 30-degree gate were selected using 65 training and 8 validation cases.
The objective was minimum pooled saved labels1–5 centroid TRE, requiring all
cases to pass and valid paired labels to remain present. Thus labels1–5
participated in cohort-level parameter selection, although not individual fits.
Earlier development experiments also used test data. These development results
must not be presented as an untouched independent-test protocol. Users should
select and freeze settings on their own development data before final testing.

Parameter defaults are dataset-specific choices, not universal accuracy claims.
The package does not use landmark labels in the single-mask registration mode.
The public source includes no patient images, masks or case-level experimental
results.

## Metrics

- Saved-label TRE: Euclidean distance between nonempty corresponding mask
  centroids, transformed by the full saved affine; channel pairs are pooled
  across cases. The metrics CLI uses channels0–5. For labels1–5 use per-case
  `channel_errors`, dropping key `0`. Empty channels are omitted, never zero-filled.
- The verification report additionally includes a **different** native
  labels1–5 TRE: original-mask centroids under the rigid matrix, averaged within
  each case then across cases. Do not confuse this with pooled saved-label TRE.
- Surfaces: centres of foreground boundary voxels from six-connected binary
  erosion, treating outside-image voxels as background, in full affine coordinates.
- Chamfer: half the sum of directional mean nearest-neighbour distances,
  Euclidean and unsquared.
- HD95: the 95th percentile of concatenated bidirectional nearest-neighbour
  distances, using NumPy's linear percentile interpolation.
- Dice: twice the foreground intersection divided by the foreground size sum.
- Surface metrics are for channel0 and aggregated across cases. Population SD
  uses ddof=0; sample SD is also stored. These SDs are not confidence intervals.
- Input spatial units in the development data were unspecified. Distances remain
  corrected world-coordinate units; physical calibration is needed to claim mm.

