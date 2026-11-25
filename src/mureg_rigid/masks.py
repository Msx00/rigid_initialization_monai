"""Register a pair of 3D binary NIfTI masks; save a forward world-coordinate transform."""
import argparse
import json
from pathlib import Path
import tempfile
import nibabel as nib
import numpy as np
from .adapter import configure_registration, configure_rotation_limit, register_masks, registration_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--moving-mask', type=Path, required=True)
    parser.add_argument('--fixed-mask', type=Path, required=True)
    parser.add_argument('--out-transform', type=Path, required=True, help='New JSON file; maps moving world coordinates to fixed world coordinates')
    parser.add_argument('--pca-rotation-gain', type=float, default=.1)
    parser.add_argument('--smoothing', type=int, default=0)
    parser.add_argument('--landmarks', type=int, default=7000)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--decimation-reduction', type=float, default=.2)
    parser.add_argument('--max-rotation-degrees', type=float, default=30.)
    args = parser.parse_args()
    if args.out_transform.exists():
        parser.error('Output transform already exists; choose a new file')
    settings = {k:getattr(args,k) for k in ('pca_rotation_gain','smoothing','landmarks','iterations','decimation_reduction')}
    try:
        configure_registration(settings)
        configure_rotation_limit(args.max_rotation_degrees)
        with tempfile.TemporaryDirectory(prefix='mureg-rigid-') as work:
            paths=[]
            for name,path in [('moving',args.moving_mask),('fixed',args.fixed_mask)]:
                image=nib.load(path)
                a=np.asanyarray(image.dataobj)
                if a.ndim!=3 or not np.isfinite(a).all() or not np.isin(np.unique(a),[0,1,255]).all() or not np.any(a):
                    raise ValueError(f'{name} mask must be a nonempty 3D binary NIfTI with values 0/1 or 0/255')
                temporary=Path(work)/(name+'.nii.gz')
                nib.save(nib.Nifti1Image((a>0).astype(np.uint8),image.affine),temporary)
                paths.append(temporary)
            result=register_masks(*paths)
    except (ValueError,TypeError,OSError) as exc:
        parser.error(str(exc))
    matrix=result.pop('T_mv2fx')
    report=dict(result,T_moving_to_fixed=matrix.tolist(),T_fixed_to_moving=np.linalg.inv(matrix).tolist(),
                parameters=registration_parameters(),coordinates='NIfTI affine world coordinates; not voxel indices')
    args.out_transform.parent.mkdir(parents=True,exist_ok=True)
    args.out_transform.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Saved {args.out_transform}; rotation={result['best_rot_angle_deg']:.3f} degrees; accepted={result['accepted']}")
    return 0 if result['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
