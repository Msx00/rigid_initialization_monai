import csv
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tcia_prostate_rigid import RegistrationConfig, register_pair
from tcia_prostate_rigid.demo import make_demo
from tcia_prostate_rigid.batch import load_manifest, run_batch
from tcia_prostate_rigid.prepare import split_manifest


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.source = cls.root / 'input'; make_demo(cls.source)
        # This grid avoids synthetic half-voxel NN ties under tiny ICP roundoff.
        cls.config = RegistrationConfig(size=(64, 64, 64), spacing_mm=(.8, .8, .8))
        cls.base = cls.call(cls.root / 'registered')

    @classmethod
    def call(cls, output, **kwargs):
        options = dict(mr_targets={1: cls.source / 'mr/target_1_label.nrrd'}, us_targets={1: cls.source / 'us/target_1_label.nrrd'}, config=cls.config)
        options.update(kwargs)
        return register_pair(cls.source / 'mr/image.nrrd', cls.source / 'us/image.nrrd',
            cls.source / 'mr/prostate_label.nrrd', cls.source / 'us/prostate_label.nrrd', output, **options)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_verified_export_and_target_metrics(self):
        result = self.base
        self.assertEqual(result.report['verification']['status'], 'passed')
        self.assertGreater(result.metrics['dice_after'], .999)
        self.assertLess(result.metrics['target_tre_after_mm'], .01)
        self.assertGreater(result.metrics['target_tre_before_mm'], 5)
        np.testing.assert_allclose(result.matrix_lps[:3, 3], [8, -6, 4], atol=.01)
        n = nib.load(result.paths['mr_image']); self.assertEqual(n.shape, (64, 64, 64))
        self.assertEqual(n.header.get_xyzt_units()[0], 'mm')
        np.testing.assert_allclose(n.affine, nib.load(result.paths['us_image']).affine)
        label = sitk.ReadImage(str(result.paths['mr_target_1']))
        self.assertEqual(set(np.unique(sitk.GetArrayViewFromImage(label))), {0, 1})

    def test_resume_preserves_outputs_and_rejects_changed_parameters(self):
        path = self.base.output_dir / 'registration.json'; before = path.stat().st_mtime_ns
        result = self.call(self.base.output_dir)
        self.assertTrue(result.resumed); self.assertEqual(path.stat().st_mtime_ns, before)
        with self.assertRaises(FileExistsError):
            self.call(self.base.output_dir, config=dataclasses.replace(self.config, iterations=25))

    def test_modified_output_is_not_silently_reused(self):
        dest = self.root / 'tampered'; shutil.copytree(self.base.output_dir, dest)
        (dest / 'mr/target_1_label.nii.gz').write_bytes(b'invalid')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.call(dest)
        rebuilt = self.call(dest, overwrite=True)
        self.assertFalse(rebuilt.resumed); self.assertEqual(rebuilt.report['verification']['status'], 'passed')

    def test_targets_do_not_change_fitted_transform(self):
        target = sitk.ReadImage(str(self.source / 'mr/target_1_label.nrrd'))
        array = np.roll(sitk.GetArrayFromImage(target), 3, axis=2)
        moved = sitk.GetImageFromArray(array); moved.CopyInformation(target)
        result = self.call(self.root / 'changed_target', mr_targets={1: moved})
        np.testing.assert_allclose(result.matrix_lps, self.base.matrix_lps, atol=1e-10)
        self.assertGreater(result.metrics['target_tre_after_mm'], 2)

    def test_input_directory_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'input'):
            self.call(self.source, overwrite=True)

    def test_unknown_nonempty_directory_is_not_replaced(self):
        dest = self.root / 'unrelated'; dest.mkdir(); (dest / 'important.txt').write_text('keep')
        with self.assertRaises(FileExistsError):
            self.call(dest, overwrite=True)
        self.assertEqual((dest / 'important.txt').read_text(), 'keep')

    def test_mismatched_mask_grid_is_rejected(self):
        im = sitk.ReadImage(str(self.source / 'mr/image.nrrd')); im.SetOrigin([100, 100, 100])
        with self.assertRaisesRegex(ValueError, 'grid'):
            register_pair(im, self.source / 'us/image.nrrd', self.source / 'mr/prostate_label.nrrd',
                          self.source / 'us/prostate_label.nrrd', self.root / 'bad_grid', config=self.config)

    def test_batch_manifest_and_missing_target_pair(self):
        rows = load_manifest(self.source / 'pairs.csv'); self.assertEqual(len(rows), 1)
        report = run_batch(self.source / 'pairs.csv', self.root / 'batch', config=self.config)
        self.assertEqual(report['state'], 'completed'); self.assertEqual(report['successful'], 1)
        resumed = run_batch(self.source / 'pairs.csv', self.root / 'batch', config=self.config)
        self.assertEqual(resumed['state'], 'completed')
        result = self.call(self.root / 'no_targets', mr_targets={}, us_targets={})
        self.assertIsNone(result.metrics['target_tre_after_mm'])
        self.assertEqual(result.metrics['target_count'], 0)

    def test_batch_error_is_recorded_without_discarding_good_case(self):
        with (self.source / 'pairs.csv').open() as f:
            row = next(csv.DictReader(f))
        bad_mask = self.source / 'empty.nrrd'; sitk.WriteImage(sitk.Image([44, 42, 40], sitk.sitkUInt8), str(bad_mask))
        bad = {**row, 'pair_id': 'synthetic-bad', 'mr_prostate_label': 'empty.nrrd'}
        path = self.root / 'mixed.csv'
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows([row, bad])
        result = run_batch(path, self.root / 'mixed_output', data_root=self.source, config=self.config)
        self.assertEqual(result['state'], 'completed_with_errors'); self.assertEqual(result['successful'], 1); self.assertEqual(result['failed'], 1)

    def test_patient_split_has_no_leakage(self):
        rows = [dict(pair_id=f'case-{i}-{j}', patient=f'subject-{i}') for i in range(10) for j in range(2)]
        path = self.root / 'split_input.csv'
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        split_manifest(path, self.root / 'splits')
        sets = []
        for split in ['train', 'val', 'test']:
            with (self.root / 'splits' / split / 'pairs.csv').open(encoding='utf-8-sig') as f:
                sets.append({r['patient'] for r in csv.DictReader(f)})
        self.assertEqual(len(set.union(*sets)), 10)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_cli_demo_and_resume(self):
        command = [sys.executable, '-m', 'tcia_prostate_rigid', 'demo', '--output', str(self.root / 'cli'), '--size', '48', '--spacing', '1']
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)['synthetic'])
        again = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(again.stdout)['resumed'])

    def test_cli_batch_with_spawn_workers(self):
        output = self.root / 'parallel'
        result = subprocess.run([sys.executable, '-m', 'tcia_prostate_rigid', 'batch',
            '--pairs', str(self.source / 'pairs.csv'), '--output', str(output), '--workers', '2',
            '--size', '64', '--spacing', '.8', '--monai'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / 'progress.json').read_text())['state'], 'completed')
        datalist = json.loads((output / 'dataset.json').read_text())
        self.assertEqual(len(datalist['test']), 1)
        self.assertEqual(datalist['training'], [])
        record = datalist['test'][0]
        for key in ('mr_image', 'us_image', 'mr_target_1', 'us_prostate_label', 'forward_transform'):
            self.assertFalse(Path(record[key]).is_absolute())
            self.assertTrue((output / record[key]).is_file())
        self.assertTrue((output / 'metrics/metric_summary.csv').is_file())

    def test_batch_cooperative_stop_and_resume(self):
        output = self.root / 'stopped_batch'; output.mkdir(); (output / 'STOP_REQUESTED').touch()
        result = run_batch(self.source / 'pairs.csv', output, config=self.config)
        self.assertEqual(result['state'], 'stopped'); self.assertEqual(result['finished'], 0)
        (output / 'STOP_REQUESTED').unlink()
        resumed = run_batch(self.source / 'pairs.csv', output, config=self.config)
        self.assertEqual(resumed['state'], 'completed')

