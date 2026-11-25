"""Lazy entry point for the optional TCIA workflow."""
import importlib.util
import sys

def main():
    missing = [name for name in ('SimpleITK', 'filelock') if importlib.util.find_spec(name) is None]
    if missing:
        print('TCIA support requires the optional dependencies. Install from the repository with: python -m pip install ".[tcia]"', file=sys.stderr)
        return 2
    from tcia_prostate_rigid.cli import main as tcia_main
    return tcia_main()
