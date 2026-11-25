"""Console entry point: tcia-rigid or python -m tcia_prostate_rigid."""
import argparse
import json
from pathlib import Path
import sys
from dataclasses import asdict
from . import __version__
from .config import RegistrationConfig


def _settings(parser):
    parser.add_argument('--config', type=Path, help='JSON settings file; explicit flags override it')
    for name, kind in [('pca-rotation-gain', float), ('smoothing', int), ('landmarks', int), ('iterations', int),
                       ('decimation-reduction', float), ('rotation-warning-degrees', float), ('threads', int)]:
        parser.add_argument('--' + name, type=kind, default=None)
    parser.add_argument('--size', type=int, nargs='+', help='One cubic size or three dimensions, default 175')
    parser.add_argument('--spacing', type=float, nargs='+', help='One isotropic spacing or three mm spacings, default 0.8')
    parser.add_argument('--assume-mm', action='store_true', help='Explicitly treat unknown NIfTI units as mm')
    parser.add_argument('--overwrite', action='store_true', help='Replace a case only after the new output passes verification')


def _config(args):
    values = json.loads(args.config.read_text()) if args.config else {}
    for key in asdict(RegistrationConfig()):
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    for option, key in [('size', 'size'), ('spacing', 'spacing_mm')]:
        v = getattr(args, option, None)
        if v is not None:
            if len(v) not in [1, 3]:
                raise ValueError(f'--{option} expects one or three values')
            values[key] = v * 3 if len(v) == 1 else v
    if args.assume_mm:
        values['assume_unknown_nifti_units_are_mm'] = True
    return RegistrationConfig(**values)


def _targets(values):
    result = {}
    for item in values or []:
        number, path = item.split('=', 1); number = int(number)
        if number in result:
            raise ValueError(f'Duplicate Target ID: {number}')
        result[number] = Path(path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='TCIA prostate MR -> US rigid registration in physical millimetres')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    pair = commands.add_parser('pair', help='Register and export one converted image/mask pair')
    for key in ['mr', 'us', 'mr-mask', 'us-mask']:
        pair.add_argument('--' + key, type=Path, required=True)
    pair.add_argument('--mr-target', action='append', metavar='ID=PATH')
    pair.add_argument('--us-target', action='append', metavar='ID=PATH')
    pair.add_argument('--output', type=Path, required=True); _settings(pair)
    batch = commands.add_parser('batch', help='Register a portable CSV pairs manifest')
    batch.add_argument('--pairs', type=Path, required=True); batch.add_argument('--data-root', type=Path)
    batch.add_argument('--output', type=Path, required=True); batch.add_argument('--workers', type=int, default=1); _settings(batch)
    batch.add_argument('--monai', action='store_true', help='Export a patient-disjoint MONAI datalist, split checks and metrics')
    convert = commands.add_parser('convert', help='Convert one TCIA DICOM series with matching STLs (requires [dicom])')
    convert.add_argument('--dicom-dir', type=Path, required=True); convert.add_argument('--modality', choices=['mr', 'us'], required=True)
    convert.add_argument('--prostate-stl', type=Path, required=True); convert.add_argument('--target-stl', action='append', metavar='ID=PATH')
    convert.add_argument('--series-uid'); convert.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('prepare', help='Prepare original TCIA metadata/workbooks/STL archive (requires [dicom])')
    prepare.add_argument('--metadata', type=Path, required=True); prepare.add_argument('--targets', type=Path, required=True)
    prepare.add_argument('--stls', type=Path, required=True); prepare.add_argument('--biopsy', type=Path)
    prepare.add_argument('--dicom-root', type=Path); prepare.add_argument('--mr-types', nargs='+', choices=['T1', 'T2'], default=['T2'])
    prepare.add_argument('--output', type=Path, required=True)
    split = commands.add_parser('split', help='Create patient-disjoint train/val/test CSVs; volumes stay in place')
    split.add_argument('--pairs', type=Path, required=True); split.add_argument('--output', type=Path, required=True)
    split.add_argument('--ratios', type=float, nargs=3, default=[.6, .1, .3]); split.add_argument('--seed', type=int, default=42)
    demo = commands.add_parser('demo', help='Generate synthetic data and run registration, no downloads needed')
    demo.add_argument('--output', type=Path, default=Path('outputs/demo')); _settings(demo)
    args = parser.parse_args(argv)
    try:
        if args.command == 'pair':
            from .pipeline import register_pair
            result = register_pair(args.mr, args.us, args.mr_mask, args.us_mask, args.output,
                mr_targets=_targets(args.mr_target), us_targets=_targets(args.us_target), config=_config(args), overwrite=args.overwrite)
            report = dict(output=str(result.output_dir), resumed=result.resumed, metrics=result.metrics, warnings=result.report['warnings'])
        elif args.command == 'batch':
            from .batch import run_batch
            report = run_batch(args.pairs, args.output, data_root=args.data_root, config=_config(args), workers=args.workers, overwrite=args.overwrite, monai=args.monai)
        elif args.command == 'convert':
            from .raw import convert_series
            report = convert_series(args.dicom_dir, args.prostate_stl, args.output, modality=args.modality,
                                    target_stls=_targets(args.target_stl), series_uid=args.series_uid)
        elif args.command == 'prepare':
            from .prepare import prepare_tcia
            report = prepare_tcia(args.metadata, args.targets, args.stls, args.output, dicom_root=args.dicom_root,
                                  biopsy_workbook=args.biopsy, mr_types=args.mr_types)
        elif args.command == 'split':
            from .prepare import split_manifest
            report = split_manifest(args.pairs, args.output, ratios=args.ratios, seed=args.seed)
        else:
            from .demo import make_demo
            from .pipeline import register_pair
            make_demo(args.output / 'input')
            source = args.output / 'input'
            result = register_pair(source / 'mr/image.nrrd', source / 'us/image.nrrd', source / 'mr/prostate_label.nrrd',
                source / 'us/prostate_label.nrrd', args.output / 'registered',
                mr_targets={1: source / 'mr/target_1_label.nrrd'}, us_targets={1: source / 'us/target_1_label.nrrd'},
                config=_config(args), overwrite=args.overwrite, case_id='synthetic-001')
            report = dict(output=str(result.output_dir), synthetic=True, resumed=result.resumed, metrics=result.metrics)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        state = report.get('state', report.get('status'))
        return 130 if state in ['interrupted', 'stopped'] else (2 if state == 'completed_with_errors' else 0)
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        print(f'tcia-rigid: {exc}', file=sys.stderr)
        return 2

