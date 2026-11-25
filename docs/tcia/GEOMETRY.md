# Physical geometry and algorithm

## Coordinate contract

All operations use **physical millimetres**, not voxel indices. Image origin, spacing and full direction matrices are retained before fitting. The public transform is column-vector `p_US_LPS = T_MR_to_US_LPS @ p_MR_LPS`.

NIfTI uses RAS affine representation. With `F = diag(-1,-1,1,1)`, `T_LPS = F @ T_RAS @ F`. This basis change does not reverse superior/inferior. ITK resampling maps each output US-world point back into the original MR using `inverse(T_LPS)`.

The output grid has identity LPS direction and is centred at the **voxel-volume centroid of the native US prostate mask**. For size175 and spacing0.8, edge-to-edge field of view is140mm; first-to-last voxel-centre span is139.2mm. Different patients have different origins. All files within a pair share one grid.

NRRD input geometry must be in mm. Known NIfTI metre/micron units are checked against ITK's mm geometry. Unknown NIfTI units are rejected unless explicitly acknowledged through `--assume-mm` / `assume_unknown_nifti_units_are_mm=True`. This option does not repair an incorrectly scaled affine.

## Fitting

1. Validate nonempty binary prostate masks and source geometry.
2. Convert their full native grids to temporary NIfTI, preserving the same float-affine precision used in the validated workflow.
3. Extract marching-cubes surfaces, optionally windowed-sinc smooth, then clean/decimate using vendored upstream primitives.
4. Compute PCA frames and enumerate24 proper signed axis correspondences. Select the one with minimum rotation angle, then multiply its rotation vector by the configured gain (default 0, so only the centroid translation is applied at initialization). Align mesh centroids.
5. Apply upstream rigid VTK ICP; compose `T = T_ICP @ T_initial`. No scale, shear or reflection is fitted.
6. Resample images linearly and labels with nearest neighbour onto the common US grid. All Target labels follow the same MR transform or fixed-US identity.

This is an adaptation, not the upstream candidate-distance scoring/multistart pipeline. PCA minimum-angle selection assumes broadly compatible input orientation. The post-fit rotation warning does not restrict the optimizer and does not establish anatomical plausibility. Increasing iterations may improve gland surfaces while worsening internal Targets.

## Eigen Artemis US and STL

TCIA Eigen US often lacks standard `ImageOrientationPatient`. The reader follows the SlicerHeart Eigen loader: isotropic spacing from `PixelAspectRatio[0]/1000` (or valid Eigen private VoxelSize fallback), scanner-centred origin, and the axis permutation used by `loadEigenArtemis3DUS`. It verifies the vendor/model and scalar multiframe dimensions. It does **not** apply an MR-style DICOM orientation interpretation to this device data.

STL coordinates are independent of the US voxel permutation: an explicit `SPACE=RAS` header is honoured; otherwise **LPS** is assumed, matching the validated Slicer convention for these files. Applying an extra US-only RAS flip misplaces labels. Nonfinite redundant triangle normals can be recomputed in a temporary copy; nonfinite vertex positions are rejected.

Matching the device convention does not independently prove patient anatomy. The S/I labels derived from headers are not extra orientation ground truth. Review gland apex/base, visible bladder neck and posterior probe relationships where identifiable.

## Verification and boundaries

Every exported NIfTI is checked for dimensions, mm units and affine equality. Saved forward/inverse ITK transforms are checked by physical-point roundtrips. Original image intensities are independently sampled using ITK point conversion plus SciPy interpolation, and prostate labels using nearest neighbour. Files receive SHA256 checksums for resume validation.

Empty resampled Targets, output gland clipping, large rotations and high Target residuals produce warnings rather than silently removing patients. Combined Target maps prefer smaller IDs in overlaps. Numerical correctness is separate from anatomical correctness.

Nearest-neighbour sampling exactly on half-voxel ties can move boundary voxels under tiny floating-point differences. Transform accuracy is therefore also checked in physical mm, rather than requiring bitwise identical masks for every grid phase.

