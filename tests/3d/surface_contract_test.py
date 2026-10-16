#!/usr/bin/env python3
"""Independent dense-cube oracle, sparse reconstruction, and file integrity tests."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import pf3d_surface as surface


def dense_oracle(phi):
    """Brute-force cube classification, without the GPU's 26-neighbour rule."""
    side = phi.shape[0]
    offsets = np.array([0, 1, side, side + 1, side**2, side**2 + 1,
                        side**2 + side, side**2 + side + 1])
    z, y, x = np.mgrid[:side-1, :side-1, :side-1]
    lower = (z * side**2 + y * side + x).ravel()
    indices = lower[:, None] + offsets[None, :]
    values = phi.ravel()[indices]
    signs = values >= np.float32(0.5)
    crossing = np.any(signs, axis=1) & ~np.all(signs, axis=1)
    return lower[crossing], values[crossing], np.unique(indices[crossing])


def neighbor_vertices(phi):
    inside = phi >= np.float32(0.5)
    padded = np.pad(inside, 1, constant_values=False)
    needed = np.zeros_like(inside)
    side = len(phi)
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                needed |= inside != padded[dz:dz+side, dy:dy+side, dx:dx+side]
    return np.flatnonzero(needed)


def sparse_cell(phi, gid=11, origin=(-20, 70, -4)):
    indices = neighbor_vertices(phi)
    data = b''.join(surface.VERTEX.pack(int(q), float(phi.ravel()[q])) for q in indices)
    return surface.Cell(gid, origin, len(phi)-2, 1.0, 0.01, 5.0,
                        (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 3, 0, data)


def make_file(cells_by_step):
    # Channel Nz16, H10, padding3, exactly matching the small GPU smoke geometry.
    values = [b'PF3SUR1\0', 1, surface.FILE.size, 1, 11, 0,
              surface.VERTEX.size, surface.CELL.size, surface.FRAME.size,
              96, 96, 16, 1., 1., 1., 0.001, 0.02, 0.5, 10, 10, 3,
              10., 1., 1., 0, 1, 1, 0]
    raw = bytearray(surface.FILE.pack(*values))
    struct.pack_into('<Q', raw, 160, zlib.crc32(raw))
    commits = []
    for step, cell in cells_by_step:
        frame = bytearray(surface.FRAME.pack(b'P3SFRM1\0', step, step * 0.001,
                                             1, cell.vertex_count, 0))
        frame += surface.CELL.pack(cell.id, *cell.origin, cell.edge, 0, cell.vertex_count,
                                   cell.gamma, cell.active_speed, cell.radius, 0.,
                                   *cell.polarity, *cell.velocity, cell.tumbles, cell.flags)
        if cell.vertex_count:
            frame += surface.CHUNK.pack(b'P3SCHK1\0', cell.vertex_count,
                                        len(cell.data), zlib.crc32(cell.data))
            frame += cell.data
        raw += frame
        raw += surface.FOOTER.pack(b'P3SCMT1\0', step, cell.vertex_count, zlib.crc32(frame))
        commits.append(len(raw))
    raw += surface.FOOTER.pack(b'P3SEND1\0', len(cells_by_step), cells_by_step[-1][0], 0)
    return raw, commits


class SurfaceContract(unittest.TestCase):
    def check_field(self, interior):
        phi = np.pad(np.asarray(interior, dtype=np.float32), 1)
        lower, values, needed = dense_oracle(phi)
        np.testing.assert_array_equal(neighbor_vertices(phi), needed)
        cell = sparse_cell(phi)
        coords, restored = surface.crossing_cubes(cell)
        side = len(phi)
        reconstructed_lower = (coords[:, 2] * side**2 + coords[:, 1] * side + coords[:, 0])
        np.testing.assert_array_equal(reconstructed_lower, lower)
        np.testing.assert_array_equal(restored.view(np.uint32), values.view(np.uint32))

    def test_dense_sparse_equivalence(self):
        self.check_field(np.zeros((8, 8, 8)))
        self.check_field(np.ones((8, 8, 8)))
        self.check_field(np.indices((8, 8, 8)).sum(axis=0) % 2)
        self.check_field(np.full((8, 8, 8), 0.5))
        field = np.zeros((8, 8, 8), dtype=np.float32)
        field[4, 4, 4] = 0.5
        self.check_field(field)
        self.assertEqual(len(neighbor_vertices(np.pad(field, 1))), 27)
        field[:] = 1
        field[3:5, 3:5, 3:5] = 0
        self.check_field(field)
        rng = np.random.default_rng(192631)
        for _ in range(20):
            field = rng.choice(np.array([-0.0, np.nextafter(np.float32(0), np.float32(1)),
                                         0.25, 0.5, 0.75, 1.0], dtype=np.float32), (8, 8, 8))
            self.check_field(field)

    def test_roundtrip_and_corruption(self):
        phi = np.zeros((10, 10, 10), dtype=np.float32)
        phi[5, 5, 5] = 0.5
        cell = sparse_cell(phi)
        raw, _ = make_file([(0, cell), (10, cell)])
        with tempfile.TemporaryDirectory(prefix='pf3d-surface-') as folder:
            path = Path(folder) / 'test.pfs3d'
            path.write_bytes(raw)
            with surface.SurfaceReader(path) as reader:
                frames = list(reader)
                self.assertTrue(reader.complete)
                self.assertEqual([f.step for f in frames], [0, 10])
                self.assertEqual(frames[-1].cells[0].data, cell.data)
            for position in [8, 64, 168, surface.FILE.size + surface.FRAME.size + surface.CELL.size + surface.CHUNK.size + 10,
                             len(raw) - surface.FOOTER.size - 1]:
                damaged = bytearray(raw)
                damaged[position] ^= 1
                path.write_bytes(damaged)
                with self.assertRaises(surface.SurfaceFormatError):
                    with surface.SurfaceReader(path, recover=True) as reader:
                        list(reader)
            path.write_bytes(raw)
            with self.assertRaises(surface.SurfaceFormatError):
                with surface.SurfaceReader(path, max_frame_bytes=1) as reader:
                    list(reader)

    def test_tail_transaction(self):
        # All possible cuts in a small second frame, plus the completion marker.
        cell = sparse_cell(np.zeros((10, 10, 10), dtype=np.float32))
        raw, commits = make_file([(0, cell), (10, cell)])
        with tempfile.TemporaryDirectory(prefix='pf3d-surface-tail-') as folder:
            path = Path(folder) / 'test.pfs3d'
            for cut in range(commits[0], len(raw)):
                path.write_bytes(raw[:cut])
                with surface.SurfaceReader(path, recover=True) as reader:
                    frames = list(reader)
                    self.assertEqual(len(frames), 1 if cut < commits[1] else 2)
                    self.assertFalse(reader.complete)
                    self.assertTrue(reader.truncated)
                with self.assertRaises(surface.IncompleteSurfaceStream):
                    with surface.SurfaceReader(path) as reader:
                        list(reader)


if __name__ == '__main__':
    unittest.main()
