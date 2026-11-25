# Portable CSV manifests

Required columns:

| Column | Meaning |
|---|---|
| `pair_id` | Unique case ID, letters/numbers/underscore/hyphen/dot |
| `mr_image`, `us_image` | Scalar3D NRRD/NIfTI paths |
| `mr_prostate_label`, `us_prostate_label` | Corresponding nonempty binary gland masks |

Optional `patient` controls grouping in summaries; default is `pair_id`. Optional `split` controls the leading output directory; original train/val/test assignments are preserved. Output is `<output>/<split>/<patient>/<pair_id>/`, omitting an empty split.

All relative paths are resolved against **`--data-root`**, defaulting to the manifest's parent. An existing split CSV whose paths start with `val/` needs the dataset root explicitly, not the `val` directory.

Two Target schemas are supported:

1. `mr_targets`, `us_targets`: JSON objects mapping IDs to paths, e.g. `{"1":"case/mr/roi.nrrd"}`. Preferred for arbitrary filenames.
2. `mr_target_labels`, `us_target_labels`: JSON arrays of paths with `target_N` in each filename, matching the existing TCIA registration_dataset schema.

Use JSON strings inside correctly CSV-quoted cells; generate them with `csv.DictWriter` rather than manual escaping. See `examples/tcia/write_manifest.py`. Independent Target masks must be binary. Only matching IDs are used for TRE; unmatched provided labels are still transformed/exported. No Target is passed to the fitting function.

Optional `mr_warnings` and `us_warnings` JSON arrays are propagated. With `batch --monai`, the optional input `status` is retained as `source_qc`. Provide a stable `patient` identifier to group repeat visits and use train, val or test as the split. Changing input contents or configuration prevents silently reusing old results. Existing processed MR files must not be passed again as native MR input.

The repository includes only a manifest template with fictional paths, not TCIA subject/UID lists. Patients without common Targets still receive images and gland metrics; Target TRE is null and excluded from TRE means.

