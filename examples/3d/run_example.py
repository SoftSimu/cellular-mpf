#!/usr/bin/env python3
"""Run the small 3D example, then continue its saved checkpoint for one step."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('smoke.args'))
    parser.add_argument('--output-dir', type=Path, default=Path('example-3d-output'))
    args = parser.parse_args()
    executable = args.executable.resolve()
    arguments = []
    for line in args.config.read_text(encoding='utf-8').splitlines():
        arguments.extend(shlex.split(line, comments=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=args.output_dir.resolve()))
    subprocess.run([str(executable), *arguments], cwd=output, check=True)
    checkpoint = output / 'checkpoints/checkpoint.pf3d'
    with checkpoint.open('rb') as source:
        header, params = source.read(96), source.read(288)
    if header[:4] != b'PF3D' or len(params) != 288:
        raise RuntimeError('missing or invalid example checkpoint')
    step = struct.unpack_from('<q', header, 24)[0]
    dt = struct.unpack_from('<d', params, 56)[0]
    subprocess.run([str(executable), '-c', str(checkpoint),
                    '--t-end', format((step + 1) * dt, '.15g'),
                    '--out', str(output / 'trajectory.txt'),
                    '--checkpoint-dir', str(output / 'checkpoints')], cwd=output, check=True)
    with checkpoint.open('rb') as source:
        resumed = source.read(96)
    if struct.unpack_from('<q', resumed, 24)[0] != step + 1:
        raise RuntimeError('checkpoint continuation did not reach its target')
    if not (output / 'trajectory.txt').stat().st_size:
        raise RuntimeError('empty example trajectory')
    print(f'GH200_3D_SMOKE_PASS output={output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
