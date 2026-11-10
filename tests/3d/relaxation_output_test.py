"""Small CPU tests of the spatial-output format and discrete integration."""
# Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
# SPDX-License-Identifier: LGPL-3.0-or-later
import io
from pathlib import Path
import struct
import sys
import unittest
import zlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import pf3d_relaxation as r


def fixture(steps=(0, 10, 17), dense_every=0):
    metadata = f'model=fixture relaxation_dense_every={dense_every}'.encode()
    raw = r.HEADER.pack(b'PF3DRLX1', 2, 18, 0, .01, 10, 64, 64, 32, 3, len(metadata), 0, 0)
    crc = zlib.crc32(metadata, zlib.crc32(raw))
    data = bytearray(raw[:72] + struct.pack('<I', crc) + raw[76:] + metadata)
    rows = np.zeros(2, dtype=r.ROW)
    rows['id'] = [5, 9007199254740993]
    rows['values'][:, 3:6] = [100, .5, 1]
    for step in steps:
        raw = r.FRAME.pack(b'P3RLFRM1', step, .01 * step, 2, 0)
        payload = rows.tobytes()
        crc = zlib.crc32(payload, zlib.crc32(raw))
        data += raw[:28] + struct.pack('<I', crc) + payload
    data += r.END.pack(b'P3RLEND1', len(steps), steps[-1])
    return data


def decode(data):
    stream = io.BytesIO(data)
    header = r.read_header(stream)
    return header, list(r.snapshots(stream, header))


class TestRelaxation(unittest.TestCase):
    def test_round_trip(self):
        header, frames = decode(fixture())
        self.assertEqual(header.box, (64, 64, 32))
        self.assertEqual([f.step for f in frames], [0, 10, 17])
        self.assertEqual(frames[0].ids[1], 9007199254740993)

    def test_corruption_and_truncation(self):
        data = fixture()
        for index in (24, 81, 115, 160, len(data)-30):
            damaged = data.copy()
            damaged[index] ^= 1
            with self.assertRaises(ValueError): decode(damaged)
        for length in (0, 7, 79, 100, len(data)-1, len(data)-24):
            with self.assertRaises(ValueError): decode(data[:length])
        with self.assertRaises(ValueError): decode(data + b'x')

    def test_missing_grid_frames(self):
        for steps in ((0, 20), (0, 7, 10), (0, 10, 10), (0, 10, 30)):
            with self.assertRaises(ValueError): decode(fixture(steps))

    def test_pre_step_quadrature(self):
        a = np.zeros((1, 18)); b = a.copy()
        a[:, r.INTERFACIAL] = [2, 3, 4]
        b[:, r.INTERFACIAL] = [5, 6, 7]
        a[:, r.ADVECTION_FACTOR] = [.9, .8]
        b[:, r.PROPULSION] = [1, 2]
        b[:, r.ADVECTION] = [2, 3]
        ids = np.array([5])
        for steps in (1, 2, 17):
            first = r.Snapshot(100, 1, ids, a)
            last = r.Snapshot(100 + steps, 1 + steps*.01, ids, b)
            delta = r.increment(first, last, .01)
            expected = .01 * sum(a[:, r.INTERFACIAL] + n/steps *
                                (b[:, r.INTERFACIAL]-a[:, r.INTERFACIAL]) for n in range(steps))
            np.testing.assert_allclose(delta['interfacial'], expected, rtol=1e-15)
            np.testing.assert_allclose(delta['advection_correction'], [[-.2, -.6, 0]])
            np.testing.assert_allclose(delta['interaction'], [[1, 1, 0]])
            reconstructed = sum(delta[k] for k in ('propulsion', 'interaction', 'interfacial',
                                                  'overlap', 'advection_correction', 'residual'))
            np.testing.assert_allclose(reconstructed, delta['displacement'], atol=1e-15)

    def test_dense_schedule(self):
        steps = tuple(range(11)) + (20,) + tuple(range(21,31)) + (37,)
        header, frames = decode(fixture(steps,dense_every=20))
        self.assertEqual([f.step for f in frames],list(steps))
        self.assertEqual(r.next_sample(7,header),8)
        self.assertEqual(r.next_sample(10,header),20)
        with self.assertRaises(ValueError): decode(fixture((0,2,10),dense_every=20))
        with self.assertRaises(ValueError): decode(fixture(dense_every=15))


if __name__ == '__main__':
    unittest.main()
