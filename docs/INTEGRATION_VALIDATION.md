# Combined release 0.2.0 validation

The combined wheel was built and installed into an isolated workspace target on
Windows with Python 3.12. All **29 synthetic tests passed**, including the original
MuReg suite and the TCIA geometry, registration, resume, patient splitting, STL,
DICOM preparation and MONAI export tests. The optional pydicom dependency was enabled.

All five installed CLI entry points passed their help checks. The documented
TCIA demo also ran with its default 175-cubed, 0.8-mm grid and completed a separate
MONAI export with a valid relative-path dataset.json.

MuReg's fitting/resampling modules are unchanged by this integration. Its package
version advances to 0.2.0 and it gains a lazy optional TCIA CLI entry point.
The TCIA registration, coordinate geometry and defaults remain those of the
supplied standalone package. One portability fix normalizes MR DICOM file paths
before comparing GDCM's list with the metadata list; previously different Windows
path separators could reject the same files. This change is recorded alongside
original and integrated hashes in TCIA-INTEGRATION.json. The original standalone
project was left unchanged. Both sets of vendored upstream primitives remain
byte-identical to their recorded source hashes.

These are software integration checks, not new clinical accuracy experiments.
The TCIA reference metrics in docs/tcia/EXPERIMENTS.md are retained from the supplied
standalone project's documentation; they were not re-estimated during integration.
Hosted Linux/Windows CI runs after pushing the combined source. The earlier
MuReg-only release checks remain in VALIDATION.md.
