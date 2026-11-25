# Data and coordinate conventions

The batch CLI accepts `.nii.gz` image pairs named identically in four folders
per requested split: `mr_images`, `us_images`, `mr_labels`, `us_labels`.
It supports the split names train, val and test. Images are XYZ or XYZ1, labels
are XYZ6 binary arrays with channel0 nonempty. Binary values must be 0/1 or
0/255; integer class maps, channels-first arrays and probability maps require
explicit conversion before loading.

Each modality's image and labels must share spatial shape and affine after
preprocessing. NIfTI affines map voxel indices to world coordinates. Axis codes
need not initially be RAS: reorientation updates both the arrays and their
affines without image interpolation.

The historical MuReg spacing correction rescales unit-length affine columns
using non-unit header voxel spacings. Ordinary affines already encoding their
spacings are preserved. This conditional rule is retained for reproducibility;
check your image metadata before using this dataset-specific correction.

Each modality is independently padded/cropped around the volume centre
to 128³. This is not a foreground-centroid crop. The affine is updated for that operation. This is not a resize or a
change in voxel spacing, and distant labels may fall outside the crop.
Verification explicitly checks that valid paired label channels survive.
Intensity normalization is per image, before resampling. The final fixed grid
is the preprocessed MR grid. US image interpolation is linear; label
interpolation is nearest-neighbour.

Forward transform direction is US world → MR world. Resampling uses the inverse
mapping from fixed-grid output voxels into the moving image. Accepted `us_images`
and `us_labels` have already had that rigid transform applied. The stored
inverse transform supports MR world → US world. Neither matrix directly maps
voxel indices unless composed with the appropriate affines.

Output image arrays are XYZ1 float32, normalized approximately to [0,1].
Output six-channel masks are XYZ6 uint8, foreground255. For channel-first
loaders, explicitly move the last axis to the front, and threshold labels >0
or divide by255. The derived argmax label maps are convenience outputs: class0
can represent both background and channel0, and overlaps are collapsed. Use
the six-channel masks for loss functions and evaluation.

The 3D mask CLI is deliberately separate: it thresholds a binary 0/1 or 0/255
mask, honours its provided affine directly, and does not apply MuReg-specific
spacing correction, 128³ crop or image normalization. Use this mode for mask
pairs already prepared in their intended world coordinates.

