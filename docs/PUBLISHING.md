# Publish your repository

Upload this repository folder only. Dataset directories and the original
experiment workspace are not part of the release. `LICENSE`, `NOTICE` and
`UPSTREAM.json` should remain in your repository.

First create an empty repository under your own GitHub account (for example,
`mureg-rigid`). Then, from this project folder, use its actual remote URL:

```bash
git init
git add .
git status --short
git commit -m "Publish adapted PCA and rigid ICP registration"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

The remote above is a placeholder. Review `git status` before committing;
the intended contents are Python source, configuration, tests and documentation.
The included `.gitignore` excludes NIfTI files, outputs and build artifacts.

After publication, users can clone the repository, run `python -m pip install .`,
then use `mureg-rigid --help`. They can also install directly with
`python -m pip install git+https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git`.

For a reproducible release, tag a tested commit (e.g. `v0.1.0`), retain the exact
environment used for your experiments, and point users to that tag. A local
versioned default profile is included as `configs/best.json`.

