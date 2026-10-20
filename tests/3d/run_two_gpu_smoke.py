#!/usr/bin/env python3
"""Compare a small channel on one/two GPUs and across a checkpoint restart."""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from pf3d_surface import SurfaceReader


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
        if not single.read_bytes() == whole.read_bytes() == resumed.read_bytes():
            raise RuntimeError('one/two-GPU or split-restart checkpoint differs')
        if (root / 'single/trajectory.txt').read_bytes() != (root / 'whole/trajectory.txt').read_bytes():
            raise RuntimeError('one/two-GPU trajectories differ')
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
