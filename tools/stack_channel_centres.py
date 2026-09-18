#!/usr/bin/env python3
"""Build fresh stacked-channel positions from passive one-layer checkpoints.

Only checkpoint metadata and cell records are read; phase fields are not copied
or CRC-validated by this position exporter. The manifest records that distinction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

from pf3d_checkpoint_header import parse_file_header


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_metadata(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    cells = []
    with path.open('rb') as stream:
        raw = stream.read(96)
        header = parse_file_header(raw)
        params = stream.read(288)
        require(len(params) == 288, 'truncated parameters')
        digest.update(raw + params)
        side, sy, nz = struct.unpack_from('<3q', params)
        height, padding = struct.unpack_from('<2q', params, 256)
        require(side == sy and side > 0 and height > 0 and
                nz == height + 2 * padding and
                struct.unpack_from('<I', params, 52)[0] == 11,
                'expected a resolved periodic-XY two-wall channel')
        radius = struct.unpack_from('<d', params, 112)[0]
        require(math.isfinite(radius) and radius > 0, 'invalid radius')
        for expected_id in range(header.num_cells):
            record = stream.read(256)
            require(len(record) == 256, 'truncated cell record')
            digest.update(record)
            gid, ox, oy, oz = struct.unpack_from('<4q', record)
            gamma, speed, cell_radius = struct.unpack_from('<3f', record, 56)
            volume, mx, my, mz, surface = struct.unpack_from('<5d', record, 72)
            tumbles, flags = struct.unpack_from('<2I', record, 140)
            edge = struct.unpack_from('<I', record, 184)[0] or header.brick_edge
            require(gid == expected_id and math.isfinite(volume) and volume > 0
                    and all(map(math.isfinite, (mx, my, mz, surface))),
                    'invalid cell identity or moments')
            require(math.isfinite(gamma) and gamma > 0 and speed == 0 and
                    cell_radius == radius and flags & ~(1 << 6) == 0,
                    'expected valid passive cells with a common radius')
            require(header.brick_edge <= edge <= 4096 and edge % 8 == 0,
                    'invalid cell storage edge')
            cells.append(dict(id=gid, x=(ox + mx / volume) % side,
                              y=(oy + my / volume) % side,
                              z=oz + mz / volume + .5 - padding,
                              gamma=gamma, speed=speed, volume=volume,
                              surface=surface, tumbles=tumbles, flags=flags,
                              edge=edge))
            stream.seek(4 * edge**3, 1)
        require(stream.tell() == before.st_size, 'incorrect checkpoint payload length')
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) ==
            (after.st_size, after.st_mtime_ns), 'checkpoint changed while reading')
    return dict(path=str(path), step=header.step, time=header.time,
                side=side, height=height, padding=padding, radius=radius,
                cells=cells, bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                stored_crc64=f'{header.stored_crc64:016x}',
                metadata_sha256=digest.hexdigest(),
                validation='metadata/layout only; phase payload not read',
                params_hex=params.hex(), base_edge=header.brick_edge,
                base_measure_shards=header.base_measure_shards)


def spacing(cells, side):
    nearest, shell = [], []
    for i, cell in enumerate(cells):
        distances = []
        for j, other in enumerate(cells):
            if i == j:
                continue
            dx = abs(cell['x'] - other['x']); dx = min(dx, side - dx)
            dy = abs(cell['y'] - other['y']); dy = min(dy, side - dy)
            distances.append(math.hypot(dx, dy))
        distances.sort()
        require(distances, 'at least two cells are required per layer')
        nearest.append(distances[0])
        count = min(6, len(distances))
        shell.append(sum(distances[:count]) / count)
    return dict(nearest_mean=sum(nearest) / len(nearest),
                nearest_min=min(nearest),
                six_neighbour_mean=sum(shell) / len(shell))


def float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def assemble(parents, seed, pitch):
    require(len(parents) >= 2, 'provide at least two distinct parent checkpoints')
    require(len({p['path'] for p in parents}) == len(parents),
            'do not clone the same parent within a stack')
    first = parents[0]
    side, radius = first['side'], first['radius']
    per_layer = len(first['cells'])
    require(math.isfinite(pitch) and pitch > 0, 'invalid layer pitch')
    height = math.ceil(2 * radius + (len(parents) - 1) * pitch)
    actual_pitch = (height - 2 * radius) / (len(parents) - 1)
    rng = random.Random(seed)
    positions, transforms, summaries = [], [], []
    for layer, parent in enumerate(parents):
        require((parent['side'], parent['radius'], len(parent['cells'])) ==
                (side, radius, per_layer), 'incompatible parent boxes/cell sizes')
        require(parent['height'] == math.ceil(2 * radius),
                'parent must be a one-layer channel')
        require(all(abs(c['z'] - parent['height'] / 2) < radius / 4
                    for c in parent['cells']), 'parent is not a single central layer')
        shift = [rng.random() * side, rng.random() * side]
        z = float32(radius + layer * actual_pitch)
        transforms.append(dict(layer=layer, shift_xy=shift, z=z,
                               source=parent['path']))
        summaries.append(spacing(parent['cells'], side))
        for cell in parent['cells']:
            x = float32((cell['x'] + shift[0]) % side) % side
            y = float32((cell['y'] + shift[1]) % side) % side
            positions.append((len(positions), x, y, z))
    minimum = math.inf
    for i, (_, x, y, z) in enumerate(positions):
        for _, qx, qy, qz in positions[:i]:
            dx = abs(x - qx); dx = min(dx, side - dx)
            dy = abs(y - qy); dy = min(dy, side - dy)
            minimum = min(minimum, math.sqrt(dx*dx + dy*dy + (z-qz)**2))
    require(minimum >= radius, 'stack violates fresh-sphere minimum separation')
    rho = len(positions) * (4 * math.pi * radius**3 / 3) / (side**2 * height)
    # Aim strictly inside the ceil-to-integer-side bin, not at its rounding edge.
    rho_argument = len(positions) * (4 * math.pi * radius**3 / 3) / ((side-.25)**2 * height)
    require(0 < rho_argument < 1, 'stack exceeds supported nominal loading')
    manifest = dict(method='translated_passive_channel_centroids', seed=seed,
                    N=len(positions), L=side, H=height, layers=len(parents),
                    radius=radius, requested_pitch=pitch, actual_pitch=actual_pitch,
                    nominal_volume_fraction=rho, rho_argument=rho_argument,
                    minimum_centre_distance=minimum, transforms=transforms,
                    parent_spacing=summaries,
                    parents=[{k:v for k,v in p.items() if k not in ('cells', 'params_hex')}
                             for p in parents],
                    transfer='positions only; fresh fields and new identity-keyed polarities')
    return positions, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--layer-pitch', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parents = [read_metadata(p) for p in args.checkpoint]
    pitch = args.layer_pitch if args.layer_pitch is not None else 2 * parents[0]['radius']
    positions, manifest = assemble(parents, args.seed, pitch)
    companion = args.output.with_suffix(args.output.suffix + '.json')
    require(not args.output.exists() and not companion.exists(), 'output already exists')
    rows = ['global_id,x,y,z', f"# channel_box={manifest['L']},{manifest['H']}"]
    rows += [f'{i},{x:.9g},{y:.9g},{z:.9g}' for i,x,y,z in positions]
    payload = ('\n'.join(rows) + '\n').encode('ascii')
    manifest['csv_sha256'] = hashlib.sha256(payload).hexdigest()
    with args.output.open('xb') as stream:
        stream.write(payload)
    with companion.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    print(json.dumps({k:manifest[k] for k in ('N','L','H','actual_pitch','nominal_volume_fraction')}))


if __name__ == '__main__':
    main()
