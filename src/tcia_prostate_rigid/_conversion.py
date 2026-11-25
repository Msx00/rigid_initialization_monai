"""TCIA conversion primitives, extracted from the locally validated conversion workflow.

Eigen geometry follows SlicerHeart DicomUltrasoundPlugin. STL uses explicit
SPACE=RAS when present and Slicer default LPS otherwise. See docs/GEOMETRY.md.
"""
import struct
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy
try:
    import pydicom
except ImportError as exc:
    pydicom = None

def affine_ras(image):
    a = np.eye(4)
    a[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    a[:3, 3] = image.GetOrigin()
    return np.diag([-1., -1., 1., 1.]) @ a

def read_us(series):
    if len(series['files']) != 1:
        raise ValueError('Expected one Eigen Artemis multiframe object')
    ds = pydicom.dcmread(series['files'][0])
    if str(ds.SeriesInstanceUID) != series['uid'] or str(ds.PatientID) != series['patient']:
        raise ValueError('DICOM patient/series mismatch')
    if str(getattr(ds, 'Manufacturer', '')) != 'Eigen' or str(getattr(ds, 'ManufacturerModelName', '')) != 'Artemis':
        raise ValueError('US conversion supports Eigen Artemis only')
    if ds.SamplesPerPixel != 1 or ds.PhotometricInterpretation != 'MONOCHROME2':
        raise ValueError('Expected scalar MONOCHROME2 Eigen data')
    private = None
    for creator in ['Eigen, Inc', 'Eigen Artemis']:
        try:
            block = ds.private_block(0x1129, creator, create=False)
            private = float(block[0x16].value)
            break
        except (KeyError, ValueError):
            pass
    standard = float(ds.PixelAspectRatio[0])/1000. if hasattr(ds, 'PixelAspectRatio') else None
    spacing = standard if standard is not None else private
    if spacing is None or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('No valid Eigen spacing; refusing to invent image geometry')
    pixels = ds.pixel_array
    if pixels.ndim != 3:
        raise ValueError('Expected a 3D Eigen multiframe pixel array')
    n_slices = int(ds.NumberOfSlices)
    if pixels.shape != (n_slices, int(ds.Rows), int(ds.Columns)):
        raise ValueError('Eigen multiframe dimensions disagree with DICOM geometry')
    image = sitk.GetImageFromArray(pixels)
    image.SetSpacing([spacing]*3)
    # Exactly P * IJKToRAS in SlicerHeart.loadEigenArtemis3DUS.
    permute = np.array([[0.,1.,0.], [0.,0.,1.], [1.,0.,0.]])
    ras_dir = permute @ np.diag([-1.,-1.,1.])
    ras_origin = permute @ np.array([.5*ds.Rows*spacing, .5*ds.Columns*spacing, -.5*n_slices*spacing])
    ras_to_lps = np.diag([-1.,-1.,1.])
    image.SetOrigin((ras_to_lps @ ras_origin).tolist())
    image.SetDirection((ras_to_lps @ ras_dir).ravel().tolist())
    info = dict(reader='pydicom + SlicerHeart Eigen Artemis geometry', pixel_spacing_standard_mm=standard,
                pixel_spacing_private_mm=private, spacing_source='PixelAspectRatio/1000' if standard is not None else 'Eigen private VoxelSize',
                spacing_conflict=standard is not None and private is not None and abs(standard-private)>1e-8)
    return image, info

def read_mr(series):
    # GDCM returns position-sorted file names, never rely on filename or InstanceNumber order.
    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series['source_dir'], series['uid']))
    # Compare normalized paths: GDCM may use forward slashes on Windows.
    if {Path(f).resolve() for f in files} != {Path(f).resolve() for f in series['files']}:
        raise ValueError('GDCM file list differs from metadata series files')
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    if str(ds.SeriesInstanceUID) != series['uid'] or str(ds.PatientID) != series['patient']:
        raise ValueError('DICOM patient/series mismatch')
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    headers = [pydicom.dcmread(f, stop_before_pixels=True, specific_tags=['ImagePositionPatient','ImageOrientationPatient','SeriesInstanceUID','PatientID']) for f in files]
    if any(str(h.SeriesInstanceUID) != series['uid'] or str(h.PatientID) != series['patient'] for h in headers):
        raise ValueError('Mixed patient or Series UID inside directory')
    positions = np.asarray([list(h.ImagePositionPatient) for h in headers], dtype=float)
    distances = positions @ normal
    gaps = np.diff(distances)
    if len(gaps) and np.min(np.abs(gaps)) < 1e-5:
        raise ValueError('Duplicate slice positions; cannot flatten multiple volumes into one')
    nonuniform = len(gaps)>1 and not np.allclose(gaps,np.median(gaps),atol=0.02,rtol=0.001)
    if any(not np.allclose(h.ImageOrientationPatient, iop, atol=1e-5) for h in headers):
        raise ValueError('Varying slice orientation')
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetDimension() != 3:
        raise ValueError('Expected a 3D MR volume')
    info = dict(reader='SimpleITK/GDCM',dicom_slice_count=len(files),slice_gap_mm=float(np.median(np.abs(gaps))) if len(gaps) else None)
    if nonuniform:
        step=float(np.median(gaps))
        indices=np.rint((distances-distances[0])/step).astype(int)
        # Recover only clearly missing planes on an otherwise regular lattice.
        # Never flatten arbitrary irregular geometry into a guessed affine.
        if step<=0 or not np.allclose(distances,distances[0]+indices*step,atol=0.02) or np.max(gaps)>5*step+.02:
            raise ValueError('Nonuniform slice spacing does not form a recoverable regular lattice')
        expected_positions=positions[0]+np.outer(indices*step,normal)
        if not np.allclose(positions,expected_positions,atol=0.02):
            raise ValueError('Nonuniform slice positions include in-plane shifts')
        missing=sorted(set(range(int(indices[-1])+1))-set(indices.tolist()))
        pixels=sitk.GetArrayFromImage(image)
        restored=np.empty((int(indices[-1])+1,*pixels.shape[1:]),dtype=pixels.dtype)
        restored[indices]=pixels
        for k in missing:
            hi=int(np.searchsorted(indices,k));lo=hi-1
            weight=(k-indices[lo])/(indices[hi]-indices[lo])
            plane=(1-weight)*pixels[lo].astype(float)+weight*pixels[hi].astype(float)
            restored[k]=np.rint(plane).astype(pixels.dtype) if np.issubdtype(pixels.dtype,np.integer) else plane
        image=sitk.GetImageFromArray(restored)
        image.SetOrigin(positions[0].tolist())
        image.SetSpacing([float(ds.PixelSpacing[1]),float(ds.PixelSpacing[0]),step])
        image.SetDirection(np.column_stack([iop[:3],iop[3:],normal]).ravel().tolist())
        info.update(missing_source_slice_indices=missing,source_slice_indices=indices.tolist(),
                    original_slice_positions_lps_mm=positions.tolist(),
                    geometry_warnings=[f'{len(missing)} missing original DICOM slice plane(s): inserted linear interpolation at z indices {missing}; all existing planes retained exactly'])
        return image,info
    # Check the complete physical slice vector including potential gantry tilt/shear.
    if len(files) > 1:
        predicted = np.asarray([image.TransformIndexToPhysicalPoint((0,0,i)) for i in range(len(files))])
        if not np.allclose(predicted, positions, atol=0.02):
            raise ValueError('ITK image geometry does not reproduce DICOM slice positions')
    return image,info

def rasterize(raw, image, modality):
    header=raw[:80].upper()
    if b'SPACE=' in header and b'SPACE=RAS' not in header and b'SPACE=LPS' not in header:
        raise ValueError('Unsupported STL SPACE convention; expected RAS or LPS')
    coordinate_system='RAS' if b'SPACE=RAS' in header else 'LPS'
    if len(raw)<84:
        raise ValueError('Truncated binary STL header')
    face_count=struct.unpack('<I',raw[80:84])[0]
    if len(raw)!=84+50*face_count:
        raise ValueError('Binary STL face count/file size mismatch')
    facet_dtype=np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attribute','<u2')])
    facets=np.frombuffer(raw,dtype=facet_dtype,offset=84,count=face_count)
    if not np.isfinite(facets['vertices']).all():
        raise ValueError('STL has non-finite vertex positions; cannot infer geometry')
    invalid_normals=~np.isfinite(facets['normal']).all(axis=1)
    invalid_normal_count=int(np.count_nonzero(invalid_normals))
    read_bytes=raw
    if invalid_normal_count:
        # Normals are redundant. Preserve every coordinate, triangle and attribute;
        # recompute only invalid stored normals in a temporary reading copy.
        sanitized=bytearray(raw)
        repaired=np.frombuffer(sanitized,dtype=facet_dtype,offset=84,count=face_count)
        vertices=facets['vertices'][invalid_normals].astype(float)
        normals=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
        lengths=np.linalg.norm(normals,axis=1)
        normals=np.divide(normals,lengths[:,None],out=np.zeros_like(normals),where=lengths[:,None]>1e-12)
        repaired['normal'][invalid_normals]=normals
        assert np.array_equal(repaired['vertices'],facets['vertices'])
        read_bytes=bytes(sanitized)
    reader = vtk.vtkSTLReader()
    with tempfile.TemporaryDirectory(prefix='tcia_stl_') as tmp:
        mesh_path = Path(tmp)/'mesh.stl'
        mesh_path.write_bytes(read_bytes)
        reader.SetFileName(str(mesh_path))
        reader.Update()
    poly = reader.GetOutput()
    if not poly.GetNumberOfPoints():
        raise ValueError(f'STL reader returned no vertices ({face_count} source triangles)')
    original_points = vtk_to_numpy(poly.GetPoints().GetData())
    ras_points = original_points * ([-1.,-1.,1.] if coordinate_system=='LPS' else [1.,1.,1.])
    bounds_ras = [ras_points.min(axis=0).tolist(), ras_points.max(axis=0).tolist()]
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    mesh_volume = vtk.vtkMassProperties()
    mesh_volume.SetInputData(poly)
    mesh_volume.Update()
    vol_mm3 = float(mesh_volume.GetVolume())
    matrix = vtk.vtkMatrix4x4()
    mesh_to_ras = np.diag([-1.,-1.,1.,1.]) if coordinate_system=='LPS' else np.eye(4)
    matrix.DeepCopy((np.linalg.inv(affine_ras(image)) @ mesh_to_ras).ravel())
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    tp = vtk.vtkTransformPolyDataFilter()
    tp.SetInputData(poly)
    tp.SetTransform(transform)
    tp.Update()
    ijk = vtk_to_numpy(tp.GetOutput().GetPoints().GetData())
    size = np.asarray(image.GetSize())
    outside = np.any((ijk < -.5) | (ijk > size-.5), axis=1)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputConnection(tp.GetOutputPort())
    stencil.SetOutputOrigin(0,0,0)
    stencil.SetOutputSpacing(1,1,1)
    stencil.SetOutputWholeExtent(0,int(size[0])-1,0,int(size[1])-1,0,int(size[2])-1)
    stencil.SetTolerance(1e-4)
    stencil.Update()
    to_image = vtk.vtkImageStencilToImage()
    to_image.SetInputConnection(stencil.GetOutputPort())
    to_image.SetInsideValue(1)
    to_image.SetOutsideValue(0)
    to_image.SetOutputScalarTypeToUnsignedChar()
    to_image.Update()
    arr = vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars()).reshape(tuple(size[::-1])).copy()
    count = int(np.count_nonzero(arr))
    if not count:
        raise ValueError(f'STL produced empty mask on reference grid; RAS bounds {bounds_ras}')
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(image)
    voxel_volume = float(np.prod(image.GetSpacing())*count)
    report = dict(stl_coordinate_system=coordinate_system, stl_coordinate_source='header' if b'SPACE=' in header else 'Slicer default LPS',
                  nonfinite_stl_normals_recomputed=invalid_normal_count,original_vertices_and_triangles_unchanged=True,
                  voxels=count, voxel_volume_mm3=voxel_volume, mesh_volume_mm3=vol_mm3,
                  voxel_to_mesh_volume_ratio=voxel_volume/vol_mm3 if vol_mm3 else None,
                  boundary_or_nonmanifold_edges=edges.GetOutput().GetNumberOfCells(),
                  fraction_mesh_vertices_outside_image=float(np.mean(outside)), mesh_bounds_ras_mm=bounds_ras,
                  ijk_bounds=[ijk.min(axis=0).tolist(),ijk.max(axis=0).tolist()])
    return mask, report
