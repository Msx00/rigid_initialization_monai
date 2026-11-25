import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import SimpleITK as sitk
import vtk
from tcia_prostate_rigid.geometry import centroid
from tcia_prostate_rigid.raw import stl_to_mask, read_dicom_series


class STLTests(unittest.TestCase):
    def test_default_lps_independent_of_modality(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sphere.STL'
            sphere = vtk.vtkSphereSource(); sphere.SetCenter(10, 15, 20); sphere.SetRadius(4)
            sphere.SetThetaResolution(30); sphere.SetPhiResolution(30); sphere.Update()
            writer = vtk.vtkSTLWriter(); writer.SetFileTypeToBinary(); writer.SetFileName(str(path)); writer.SetInputConnection(sphere.GetOutputPort()); writer.Write()
            ref = sitk.Image([64] * 3, sitk.sitkUInt8); ref.SetSpacing([.7, .8, .9]); ref.SetOrigin([-10, -12, -9])
            a, qa = stl_to_mask(path, ref, modality='mr'); b, _ = stl_to_mask(path, ref, modality='us')
            np.testing.assert_array_equal(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b))
            self.assertLess(np.linalg.norm(centroid(a) - [10, 15, 20]), .2)
            self.assertEqual(qa['stl_coordinate_system'], 'LPS')
            raw = bytearray(path.read_bytes()); raw[:80] = b'SPACE=RAS'.ljust(80, b' ')
            # Convert vertex positions to RAS before claiming a RAS header.
            dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
            facets = np.frombuffer(raw, dtype=dtype, offset=84); facets['vertices'] *= [-1, -1, 1]
            c, qc = stl_to_mask(bytes(raw), ref, modality='us')
            self.assertEqual(qc['stl_coordinate_system'], 'RAS')
            np.testing.assert_array_equal(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(c))


@unittest.skipUnless(importlib.util.find_spec('pydicom'), 'optional pydicom not installed')
class DicomTests(unittest.TestCase):
    def dataset(self, path, modality, uid):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UltrasoundMultiFrameImageStorage, generate_uid
        meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage if modality == 'MR' else UltrasoundMultiFrameImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientID = 'SYNTHETIC'; ds.SeriesInstanceUID = uid; ds.StudyInstanceUID = generate_uid(); ds.Modality = modality
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'; ds.PixelRepresentation = 0
        ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15
        return ds

    def test_eigen_pixels_and_axis_permutation(self):
        from pydicom.uid import generate_uid
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'us.dcm'; ds = self.dataset(path, 'US', generate_uid())
            ds.Manufacturer = 'Eigen'; ds.ManufacturerModelName = 'Artemis'
            ds.Rows = 12; ds.Columns = 12; ds.NumberOfFrames = 8; ds.NumberOfSlices = 8; ds.PixelAspectRatio = [300, 300]
            pixels = np.arange(8 * 12 * 12, dtype=np.uint16).reshape(8, 12, 12); ds.PixelData = pixels.tobytes(); ds.save_as(path, enforce_file_format=True)
            image, info = read_dicom_series(temp, 'us')
            np.testing.assert_array_equal(sitk.GetArrayViewFromImage(image), pixels)
            np.testing.assert_allclose(image.GetSpacing(), [.3] * 3)
            np.testing.assert_allclose(image.GetOrigin(), [-1.8, 1.2, 1.8])
            np.testing.assert_allclose(np.asarray(image.GetDirection()).reshape(3, 3), [[0, 1, 0], [0, 0, -1], [-1, 0, 0]])

    def test_mr_missing_plane_interpolates_and_records_warning(self):
        from pydicom.uid import generate_uid
        with tempfile.TemporaryDirectory() as temp:
            uid = generate_uid(); study = generate_uid()
            for i, z in enumerate([0, 2, 6, 8]):
                path = Path(temp) / f'{9-i}.dcm'; ds = self.dataset(path, 'MR', uid); ds.StudyInstanceUID = study
                ds.Rows = 8; ds.Columns = 8; ds.InstanceNumber = 9 - i
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.ImagePositionPatient = [3, -7, z]
                ds.PixelSpacing = [1.2, .9]; ds.SliceThickness = 2; ds.SpacingBetweenSlices = 2
                ds.PixelData = np.full((8, 8), z * 10, np.uint16).tobytes(); ds.save_as(path, enforce_file_format=True)
            image, info = read_dicom_series(temp, 'mr')
            self.assertEqual(image.GetSize(), (8, 8, 5))
            np.testing.assert_allclose(image.GetSpacing(), [.9, 1.2, 2])
            self.assertEqual(info['missing_source_slice_indices'], [2])
            np.testing.assert_array_equal(sitk.GetArrayViewFromImage(image)[2], np.full((8, 8), 40))

    def test_raw_collection_preparation_end_to_end(self):
        import csv, json
        from zipfile import ZipFile
        from xml.etree import ElementTree as ET
        from pydicom.uid import generate_uid
        from tcia_prostate_rigid.prepare import prepare_tcia
        from tcia_prostate_rigid.batch import load_manifest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / 'raw'; raw.mkdir(); stls = root / 'stls'; stls.mkdir()
            mr_uid, us_uid = generate_uid(), generate_uid()
            for mod, uid in [('MR', mr_uid), ('US', us_uid)]:
                folder = raw / mod.lower(); folder.mkdir(); study = generate_uid()
                for i, z in enumerate([0, 2, 6, 8] if mod == 'MR' else [0]):
                    path = folder / f'{i}.dcm'; ds = self.dataset(path, mod, uid); ds.StudyInstanceUID = study
                    if mod == 'MR':
                        ds.Rows = 8; ds.Columns = 8; ds.InstanceNumber = i + 1
                        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.ImagePositionPatient = [3, -7, z]
                        ds.PixelSpacing = [1.2, .9]; ds.SliceThickness = 2
                        ds.PixelData = np.full((8, 8), z * 10, np.uint16).tobytes()
                    else:
                        ds.Manufacturer = 'Eigen'; ds.ManufacturerModelName = 'Artemis'
                        ds.Rows = 12; ds.Columns = 12; ds.NumberOfFrames = 8; ds.NumberOfSlices = 8; ds.PixelAspectRatio = [300, 300]
                        ds.PixelData = np.ones((8, 12, 12), np.uint16).tobytes()
                    ds.save_as(path, enforce_file_format=True)
                for kind in ['ProstateSurface', 'Target1']:
                    sphere = vtk.vtkSphereSource(); sphere.SetThetaResolution(20); sphere.SetPhiResolution(20)
                    sphere.SetCenter(*([6, -3, 4] if mod == 'MR' else [0, 0, 0]))
                    sphere.SetRadius((2 if kind == 'ProstateSurface' else 1) if mod == 'MR' else (.65 if kind == 'ProstateSurface' else .4)); sphere.Update()
                    writer = vtk.vtkSTLWriter(); writer.SetFileTypeToBinary()
                    writer.SetFileName(str(stls / f'SYNTHETIC-{kind}-seriesUID-{uid}.STL'))
                    writer.SetInputConnection(sphere.GetOutputPort()); writer.Write()
            with (raw / 'metadata.csv').open('w', newline='') as f:
                fields = ['Series UID', 'Subject ID', 'Modality', 'Series Description', 'File Location', 'Number of Images']
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                for mod, uid, count in [('MR', mr_uid, 4), ('US', us_uid, 1)]:
                    w.writerow(dict(zip(fields, [uid, 'SYNTHETIC', mod, 'T2 synthetic' if mod == 'MR' else 'US', mod.lower(), count])))
            namespace = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
            sheet = ET.Element('worksheet', xmlns=namespace); data = ET.SubElement(sheet, 'sheetData')
            for number, values in enumerate([['Patient ID', 'seriesInstanceUID_MR', 'seriesInstanceUID_US', 'Target No.'], ['SYNTHETIC', mr_uid, us_uid, '1']], 1):
                row = ET.SubElement(data, 'row', r=str(number))
                for column, value in zip('ABCD', values):
                    cell = ET.SubElement(row, 'c', r=f'{column}{number}', t='inlineStr')
                    ET.SubElement(ET.SubElement(cell, 'is'), 't').text = value
            with ZipFile(root / 'targets.xlsx', 'w') as z:
                z.writestr('xl/worksheets/sheet1.xml', ET.tostring(sheet))
            prepared = root / 'prepared'
            report = prepare_tcia(raw / 'metadata.csv', root / 'targets.xlsx', stls, prepared)
            self.assertEqual(report['prepared_pairs'], 1); self.assertEqual(report['converted_series'], 2)
            cases = load_manifest(prepared / 'pairs.csv'); self.assertEqual(len(cases), 1)
            self.assertEqual(set(cases[0]['mr_targets']), {1}); self.assertEqual(set(cases[0]['us_targets']), {1})
            repeated = prepare_tcia(raw / 'metadata.csv', root / 'targets.xlsx', stls, prepared)
            self.assertEqual(repeated['prepared_pairs'], 1)

