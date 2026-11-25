"""Small synthetic functional tests; no external data or network required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np
from mureg_rigid import adapter
from mureg_rigid.run import seg_to_onehot_6


def ellipsoid():
    x,y,z=np.indices((32,32,32))
    return (((x-15)/9)**2+((y-16)/6)**2+((z-14)/4)**2<=1).astype(np.uint8)


class Workflows(unittest.TestCase):
    def setUp(self):
        self.work=tempfile.TemporaryDirectory()
        self.root=Path(self.work.name)
        adapter.configure_registration(dict(pca_rotation_gain=.1,smoothing=0,landmarks=7000,iterations=30,decimation_reduction=.2))
        adapter.configure_rotation_limit(30)

    def tearDown(self):
        self.work.cleanup()

    def masks(self):
        fixed=self.root/'fixed.nii.gz'; moving=self.root/'moving.nii.gz'
        affine=np.eye(4); affine[:3,3]=[-4,2,-3]
        nib.save(nib.Nifti1Image(ellipsoid(),np.eye(4)),fixed)
        nib.save(nib.Nifti1Image(ellipsoid(),affine),moving)
        return moving,fixed

    def invoke(self,*args):
        result=subprocess.run([sys.executable,'-B',*map(str,args)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return result

    def test_world_coordinate_transform_and_zero_gain(self):
        moving,fixed=self.masks()
        # Disable simplification to keep this analytic translation fixture exact.
        adapter.configure_registration(dict(pca_rotation_gain=0.,smoothing=0,landmarks=7000,iterations=30,decimation_reduction=0.))
        result=adapter.register_masks(moving,fixed)
        np.testing.assert_allclose(result['T_mv2fx'][:3,3],[4,-2,3],atol=1e-5)
        np.testing.assert_allclose(result['attempts'][0]['initial_matrix'][:3],np.array([[1,0,0,4],[0,1,0,-2],[0,0,1,3]]),atol=1e-5)
        self.assertEqual(result['attempts'][0]['init_type'],'centroid_only')

    def test_binary_encoding_validation(self):
        a=np.zeros((8,8,8,6),np.uint8);a[2:6,2:6,2:6,0]=1
        np.testing.assert_array_equal(seg_to_onehot_6(a),seg_to_onehot_6(a*255))
        with self.assertRaises(ValueError):seg_to_onehot_6(a.astype(float)+.2)
        with self.assertRaises(ValueError):seg_to_onehot_6(a[...,0])

    def test_mask_cli(self):
        moving,fixed=self.masks(); destination=self.root/'transform.json'
        self.invoke('-m','mureg_rigid.masks','--moving-mask',moving,'--fixed-mask',fixed,'--out-transform',destination,'--decimation-reduction',0)
        result=json.loads(destination.read_text())
        self.assertTrue(result['accepted'])
        np.testing.assert_allclose(np.array(result['T_moving_to_fixed'])[:3,3],[4,-2,3],atol=1e-5)
        self.assertIn('pca_frames',result['attempts'][0])

    def test_batch_verify_and_metrics(self):
        data=self.root/'input';out=self.root/'output';mask=ellipsoid()
        labels=np.zeros((*mask.shape,6),np.uint8);labels[...,0]=mask
        labels[14:17,15:18,13:16,1]=1
        for mod in ('mr','us'):
            affine=np.eye(4)
            if mod=='us':affine[:3,3]=[-4,2,-3]
            for kind,array in [('images',mask.astype(np.float32)),('labels',labels)]:
                f=data/'test'/f'{mod}_{kind}'/'synthetic.nii.gz';f.parent.mkdir(parents=True,exist_ok=True)
                nib.save(nib.Nifti1Image(array,affine),f)
        self.invoke('-m','mureg_rigid','--data-root',data,'--out-dir',out,'--subsets','test','--decimation-reduction',0)
        self.invoke('-m','mureg_rigid.verify','--data-root',data,'--out-dir',out,'--subsets','test')
        self.invoke('-m','mureg_rigid.metrics','--out-dir',out,'--subset','test')
        r=json.loads((out/'test_four_metrics.json').read_text())
        self.assertEqual(r['cases'],1)
        self.assertEqual(r['metrics']['TRE']['count'],2)
        self.assertAlmostEqual(r['metrics']['Dice']['mean'],1.)
        self.assertLess(r['metrics']['TRE']['mean'],1e-5)
        m=json.loads((out/'test/meta/synthetic.json').read_text())
        np.testing.assert_allclose(np.array(m['T_us2mr_4x4'])[:3,3],[4,-2,3],atol=1e-5)


if __name__=='__main__':unittest.main()
