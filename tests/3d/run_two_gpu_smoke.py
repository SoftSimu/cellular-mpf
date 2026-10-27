#!/usr/bin/env python3
"""Compare a small channel on one/two GPUs and across a checkpoint restart."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from pf3d_surface import SurfaceReader
from pf3d_checkpoint_header import canonical_crc_header, parse_file_header


def close_values(left, right, *, single_precision=False):
    # Short-run roundoff allowance, not a long-trajectory acceptance criterion.
    # Device partitioning can select different, algebraically equivalent kernels.
    tolerance = 8 * 2**-23 if single_precision else 32 * 2**-52
    floor = 8 * 2**-149 if single_precision else 32 * 2**-1074
    return len(left) == len(right) and all(
        math.isfinite(a) and math.isfinite(b)
        and abs(a - b) <= floor + tolerance * max(abs(a), abs(b))
        for a, b in zip(left, right))


def compare_checkpoints(single, two):
    left, right = single.read_bytes(), two.read_bytes()
    header = parse_file_header(left[:96])
    parse_file_header(right[:96])
    if (len(left) != len(right)
            or canonical_crc_header(left[:96]) != canonical_crc_header(right[:96])
            or left[96:384] != right[96:384]):
        raise RuntimeError('one/two-GPU checkpoint metadata differs')
    offset = 384
    for _ in range(header.num_cells):
        a, b = left[offset:offset+256], right[offset:offset+256]
        # IDs, origins, polarity, parameters, support state and counters are exact.
        if any(a[lo:hi] != b[lo:hi] for lo, hi in ((0, 44), (56, 68), (136, 256))):
            raise RuntimeError('one/two-GPU discrete or polarity state differs')
        for lo, hi, kind in ((44, 56, 'f'), (68, 72, 'f'), (72, 136, 'd')):
            x = [v[0] for v in struct.iter_unpack('<' + kind, a[lo:hi])]
            y = [v[0] for v in struct.iter_unpack('<' + kind, b[lo:hi])]
            if not close_values(x, y, single_precision=kind == 'f'):
                raise RuntimeError('one/two-GPU measured state exceeds roundoff allowance')
        edge = struct.unpack_from('<I', a, 184)[0]
        offset += 256
        size = 4 * edge**3
        x = [v[0] for v in struct.iter_unpack('<f', left[offset:offset+size])]
        y = [v[0] for v in struct.iter_unpack('<f', right[offset:offset+size])]
        if not close_values(x, y, single_precision=True):
            raise RuntimeError('one/two-GPU phase fields exceed roundoff allowance')
        offset += size
    if offset != len(left):
        raise RuntimeError('unexpected checkpoint payload length')


def compare_trajectories(single, two):
    a, b = single.read_text().splitlines(), two.read_text().splitlines()
    if len(a) != len(b):
        raise RuntimeError('one/two-GPU trajectory lengths differ')
    for left, right in zip(a, b):
        if left.startswith('#') or right.startswith('#'):
            if left != right:
                raise RuntimeError('one/two-GPU trajectory headers differ')
            continue
        x, y = left.split(), right.split()
        if len(x) != 21 or len(y) != 21:
            raise RuntimeError('unexpected trajectory columns')
        for column in range(21):
            if column in (0, 1, 8, 9, 10, 11, 12):
                if x[column] != y[column]:
                    raise RuntimeError('one/two-GPU time, ID, polarity or parameters differ')
            elif not close_values([float(x[column])], [float(y[column])],
                                  single_precision=column in (5, 6, 7, 16)):
                raise RuntimeError('one/two-GPU trajectory exceeds roundoff allowance')


def recorded_frames(path):
    result = []
    with SurfaceReader(path) as source:
        for frame in source:
            result.append((frame.step, tuple(
                (cell.id, cell.origin, cell.polarity, cell.tumbles,
                 tuple(sorted((index, struct.pack('<f', value)) for index, value in cell.samples())))
                for cell in sorted(frame.cells, key=lambda item: item.id))))
        if not source.complete:
            raise RuntimeError('incomplete surface stream')
    return result


def run(executable, directory, target, *, two=False, checkpoint=None):
    directory.mkdir()
    if checkpoint is None:
        arguments = ['--geometry', 'channel', '--channel-height', '8', '--N', '4',
                     '--radius', '4', '--lambda', '2', '--rho', '.02', '--brick-edge', '48',
                     '--kappa', '.1', '--wall-kappa', '.1', '--dt', '.01',
                     '--tau', '.015', '--v-A', '.01', '--aging-time', '0',
                     '--seed', '1234', '--polarity-seed', '5678', '--measure-shards', '1']
    else:
        arguments = ['-c', str(checkpoint)]
    arguments += ['--t-end', str(target), '--memory-mode', 'throughput',
                  '--out', str(directory / 'trajectory.txt'), '--trajectory-interval', '1',
                  '--checkpoint-dir', str(directory / 'checkpoints'), '--checkpoint-interval', '1',
                  '--surface-out', str(directory / 'surface.pfs3d'), '--surface-interval', '1',
                  '--surface-compression', 'none', '--print-interval', '0']
    if two:
        arguments += ['--device', '0', '--peer-device', '1']
    subprocess.run([str(executable), *arguments], cwd=directory, check=True)
    return directory / 'checkpoints/checkpoint.pf3d'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    args = parser.parse_args()
    reference, executable = args.reference.resolve(), args.executable.resolve()
    with tempfile.TemporaryDirectory(prefix='pf3d-two-gpu-smoke-') as temporary:
        root = Path(temporary)
        single = run(reference, root / 'single', .05)
        whole = run(executable, root / 'whole', .05, two=True)
        first = run(executable, root / 'first', .02, two=True)
        resumed = run(executable, root / 'resumed', .05, two=True, checkpoint=first)
        if whole.read_bytes() != resumed.read_bytes():
            raise RuntimeError('split-restart checkpoint differs')
        compare_checkpoints(single, whole)
        compare_trajectories(root / 'single/trajectory.txt', root / 'whole/trajectory.txt')
        single_frames = recorded_frames(root / 'single/surface.pfs3d')
        whole_frames = recorded_frames(root / 'whole/surface.pfs3d')
        first_frames = recorded_frames(root / 'first/surface.pfs3d')
        resumed_frames = recorded_frames(root / 'resumed/surface.pfs3d')
        if single_frames != whole_frames or whole_frames != first_frames[:-1] + resumed_frames:
            raise RuntimeError('one/two-GPU or split-restart interface samples differ')
        if not any(cell[3] for _, cells in whole_frames for cell in cells):
            raise RuntimeError('smoke test did not exercise a tumble')
    print('GH200_TWO_GPU_SMOKE_PASS')


if __name__ == '__main__':
    main()
