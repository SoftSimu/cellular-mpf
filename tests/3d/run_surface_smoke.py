#!/usr/bin/env python3
"""Run synthetic GPU output fixtures and compare to an independent dense oracle."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

from surface_contract_test import dense_oracle
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from pf3d_surface import SurfaceReader, crossing_cubes


def verify_fixtures(folder):
    checked = 0
    for test in range(4):
        root = Path(folder) / f'fixture-{test}'
        with SurfaceReader(root.with_suffix('.pfs3d')) as reader:
            frames = list(reader)
            if not reader.complete or len(frames) != 1:
                raise ValueError('synthetic fixture must have one complete frame')
            cells = {cell.id: cell for cell in frames[0].cells}
        with root.with_suffix('.dense').open('rb') as stream:
            count, = struct.unpack('<I', stream.read(4))
            if count != len(cells):
                raise ValueError('incorrect synthetic fixture population')
            for _ in range(count):
                gid, edge, ox, oy, oz, words = struct.unpack('<qI3qQ', stream.read(44))
                side = edge + 2
                if words != side**3:
                    raise ValueError('invalid dense oracle extent')
                phi = np.frombuffer(stream.read(words * 4), dtype='<f4').reshape((side, side, side))
                cell = cells.pop(gid)
                if cell.edge != edge or cell.origin != (ox, oy, oz):
                    raise ValueError('persistent identity or unwrapped origin mismatch')
                lower, expected_values, indices = dense_oracle(phi)
                observed_indices, observed_values = cell.arrays()
                np.testing.assert_array_equal(observed_indices, indices)
                np.testing.assert_array_equal(observed_values.view(np.uint32), phi.ravel()[indices].view(np.uint32))
                coordinates, restored = crossing_cubes(cell)
                observed_lower = (coordinates[:, 2] * side**2 + coordinates[:, 1] * side + coordinates[:, 0])
                np.testing.assert_array_equal(observed_lower, lower)
                np.testing.assert_array_equal(restored.view(np.uint32), expected_values.view(np.uint32))
                checked += 1
            if stream.read(1) or cells:
                raise ValueError('unconsumed dense fixture data')
    invalid = Path(folder) / 'nonfinite.pfs3d'
    if invalid.exists():
        with SurfaceReader(invalid, recover=True) as reader:
            if list(reader) or reader.complete or not reader.truncated:
                raise ValueError('non-finite fixture was incorrectly committed')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--fixtures', type=Path)
    args = parser.parse_args()
    if args.fixtures:
        checked = verify_fixtures(args.fixtures)
    elif args.executable:
        with tempfile.TemporaryDirectory(prefix='pf3d-surface-gpu-') as root:
            output = Path(root) / 'fixtures'
            result = subprocess.run([str(args.executable.resolve()), str(output)])
            if result.returncode == 77:
                return 77
            result.check_returncode()
            checked = verify_fixtures(output)
    else:
        parser.error('--executable or --fixtures is required')
    print(f'SURFACE_DENSE_ORACLE_PASS: {checked} cell fields, all crossing corners bit-exact')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
