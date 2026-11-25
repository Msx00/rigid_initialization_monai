> Historical MuReg-only checks. The combined 0.2.0 validation is recorded separately.

# MuReg 0.1.0 release validation

Version 0.1.0 was built as a wheel and installed into an isolated target directory
using Python 3.12 on Windows. Dependencies matched `requirements-tested.txt`.

The following synthetic functional tests passed:

1. Known world-coordinate translation with zero-gain centroid initialization.
2. Equivalent 0/1 and 0/255 label encodings, with invalid label formats rejected.
3. The 3D mask CLI, including transform and PCA-frame diagnostic output.
4. Batch registration followed by verification and metric evaluation on synthetic
   paired images and six-channel labels.

Two existing development cases were also processed with the installed release's
default profile. Both forward rigid matrices matched the saved reference exactly
(maximum absolute matrix difference 0). All seven output NIfTI arrays and their
affines matched for both cases. This checks packaging compatibility, not clinical
accuracy or independent generalization.

All four installed command-line entry points passed their help invocation checks.
The vendored upstream file hashes matched `UPSTREAM.json`. The included CI workflow
is configured for Python 3.10 on Linux and Python 3.12 on Windows; those hosted
jobs have not been executed locally and will run after the repository is pushed.

No real data, case-level records or local experiment outputs are included in the
source release. Re-run the synthetic tests after changing dependencies or code.
