"""Recorded interfaces to standard meshes, including periodic placement and IDs."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import pf3d_export_surfaces as exporter
from pf3d_contact import mesh_from_surface
from pf3d_surface import SurfaceReader
from contact_cli_test import stream_bytes
from contact_geometry_test import cell_from_dense, sphere_phi


def read_ply(path):
    with path.open('rb') as stream:
        lines = []
        while not lines or lines[-1] != 'end_header':
            lines.append(stream.readline().decode('ascii').strip())
        n = int(next(s for s in lines if s.startswith('element vertex ')).split()[-1])
        m = int(next(s for s in lines if s.startswith('element face ')).split()[-1])
        vertices = np.frombuffer(stream.read(n * 24), dtype='<f8').reshape(-1, 3)
        faces = np.frombuffer(stream.read(m * 13), dtype=[('count','u1'),('vertices','<i4',(3,))])
        if stream.read(1) or np.any(faces['count'] != 3):
            raise ValueError('invalid PLY triangle payload')
    return vertices, faces['vertices']


class SurfaceExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pf3d-export-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = cell_from_dense(sphere_phi(radius=3.), gid=2, origin=(2,10,10))
        self.b = replace(self.a, id=9, origin=(91,10,10))
        self.source = self.root / 'surface.pfs3d'
        self.source.write_bytes(stream_bytes([(s, (self.a,self.b)) for s in (0,10,20)]))

    def run_export(self, *args, target='out', expect=0):
        target = self.root / target
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as error:
            code = exporter.main([str(a) for a in (*args, '--output-dir', target)])
        self.assertEqual(code, expect, error.getvalue())
        return target

    def test_selected_meshes_preserve_double_coordinates_faces_and_metadata(self):
        out = self.run_export(self.source, '--steps', 0, 20, '--ids', 9)
        report = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(report['state'], 'PASS')
        self.assertEqual([f['step'] for f in report['frames']], [0,20])
        with SurfaceReader(self.source) as reader:
            frame = next(iter(reader)); mesh = mesh_from_surface(frame.cells[1], reader.header)
        for frame in report['frames']:
            cell, = frame['cells']
            self.assertEqual((cell['id'], cell['tumble_count']), (9, self.b.tumbles))
            vertices, faces = read_ply(out / cell['file'])
            np.testing.assert_array_equal(vertices, mesh.vertices)
            np.testing.assert_array_equal(faces, mesh.faces)
        self.assertTrue(report['streams'][0]['complete'])
        self.run_export(self.source, '--steps', 0, expect=2)

    def test_periodic_alignment_moves_the_whole_cell_only(self):
        out = self.run_export(self.source, '--steps', 0, '--anchor-id', 2)
        cells = json.loads((out / 'manifest.json').read_text())['frames'][0]['cells']
        self.assertEqual(cells[0]['image_shift'], [0.,0.,0.])
        self.assertEqual(cells[1]['image_shift'], [-96.,0.,0.])
        a, _ = read_ply(out / cells[0]['file']); b, _ = read_ply(out / cells[1]['file'])
        self.assertAlmostEqual(float(a.mean(axis=0)[0] - b.mean(axis=0)[0]), 7.)

    def test_identical_restart_endpoint_is_deduplicated(self):
        other = self.root / 'next.pfs3d'
        other.write_bytes(stream_bytes([(20, (self.b,self.a)), (30, (self.a,self.b))]))
        out = self.run_export(self.source, other, '--steps', 20, 30)
        report = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(len(report['frames']), 2)
        self.assertTrue(all(s['complete'] for s in report['streams']))
        other.write_bytes(stream_bytes([(20, (self.a,replace(self.b,tumbles=100)))]))
        out = self.run_export(self.source, other, '--steps', 20, target='changed', expect=2)
        self.assertEqual(json.loads((out / 'manifest.json').read_text())['state'], 'FAIL')

    def test_missing_frames_cells_or_incomplete_stream_fail_closed(self):
        for name, args in [('step', ('--steps',30)), ('id', ('--steps',0,'--ids',100))]:
            out = self.run_export(self.source, *args, target=name, expect=2)
            self.assertEqual(json.loads((out / 'manifest.json').read_text())['state'], 'FAIL')
        self.source.write_bytes(self.source.read_bytes()[:-5])
        out = self.run_export(self.source, '--steps', 0, target='truncated', expect=2)
        self.assertEqual(json.loads((out / 'manifest.json').read_text())['state'], 'FAIL')

    def test_independent_vtk_reader(self):
        try:
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy
        except ImportError:
            self.skipTest('optional VTK reader not installed')
        out = self.run_export(self.source, '--steps', 0, '--ids', 2)
        cell = json.loads((out / 'manifest.json').read_text())['frames'][0]['cells'][0]
        path = out / cell['file']
        reader = vtk.vtkPLYReader(); reader.SetFileName(str(path))
        reader.Update()
        vertices, faces = read_ply(path)
        decoded = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
        # VTK's PLY reader converts point coordinates to binary32 internally;
        # the independent binary reader above checks lossless binary64 storage.
        np.testing.assert_array_equal(decoded, vertices.astype(decoded.dtype))
        self.assertEqual(reader.GetOutput().GetNumberOfCells(), len(faces))


if __name__ == '__main__':
    unittest.main()
