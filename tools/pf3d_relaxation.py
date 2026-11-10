#!/usr/bin/env python3
"""Read independent 3D relaxation rates and integrate accepted-step intervals."""
# Copyright (C) 2026 Steven A. Silber and Mikko Karttunen
# SPDX-License-Identifier: LGPL-3.0-or-later
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import struct
import zlib

import numpy as np

HEADER = struct.Struct('<8sIIQdQQQQIIII')
FRAME = struct.Struct('<8sQdII')
END = struct.Struct('<8sQQ')
ROW = np.dtype([('id', '<i8'), ('values', '<f8', (18,))])
CENTROID = slice(0, 3)
VOLUME, MOBILITY, GAMMA = 3, 4, 5
PROPULSION, ADVECTION = slice(6, 8), slice(8, 10)
INTERFACIAL, OVERLAP, ADVECTION_FACTOR = slice(10, 13), slice(13, 16), slice(16, 18)


@dataclass(frozen=True)
class Header:
    cells: int
    start_step: int
    dt: float
    interval: int
    box: tuple
    boundary_flags: int
    metadata: str
    dense_every: int = 0


@dataclass(frozen=True)
class Snapshot:
    step: int
    time: float
    ids: np.ndarray
    values: np.ndarray


def _read(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError('Truncated relaxation segment; no data are imputed.')
    return value


def read_header(stream):
    raw = _read(stream, HEADER.size)
    magic, cells, values, start, dt, interval, nx, ny, nz, flags, size, crc, reserved = HEADER.unpack(raw)
    if (magic != b'PF3DRLX1' or not 0 < cells <= 1_000_000 or values != 18 or
            not np.isfinite(dt) or dt <= 0 or not interval or min(nx, ny, nz) <= 0 or
            flags not in (3, 19, 35) or not 0 < size <= 65536 or reserved):
        raise ValueError('Invalid or unsupported relaxation header.')
    metadata = _read(stream, size)
    checksum = zlib.crc32(raw[:72] + bytes(4) + raw[76:])
    if zlib.crc32(metadata, checksum) != crc:
        raise ValueError('Relaxation header checksum mismatch.')
    text = metadata.decode('utf-8')
    tokens = [word.split('=',1)[1] for word in text.split() if word.startswith('relaxation_dense_every=')]
    if len(tokens) > 1:
        raise ValueError('Duplicate dense-sampling metadata.')
    dense_every = int(tokens[0]) if tokens else 0
    if dense_every < 0 or dense_every % interval:
        raise ValueError('Dense period must be a multiple of the sampling interval.')
    return Header(cells, start, dt, interval, (nx, ny, nz), flags, text, dense_every)


def next_sample(step, header):
    if header.dense_every and step % header.dense_every < header.interval:
        return step + 1
    return (step // header.interval + 1) * header.interval


def on_grid(step, header):
    return step % header.interval == 0 or (header.dense_every and
                                         step % header.dense_every <= header.interval)


def snapshots(stream, header):
    """Yield validated frames. Exhaust the iterator to verify the completion record."""
    count, last, identity = 0, None, None
    while True:
        magic = _read(stream, 8)
        if magic == b'P3RLEND1':
            _, frames, step = END.unpack(magic + _read(stream, END.size - 8))
            if not count or frames != count or step != last or stream.read(1):
                raise ValueError('Invalid relaxation completion record.')
            return
        if magic != b'P3RLFRM1':
            raise ValueError('Invalid relaxation frame magic.')
        raw = magic + _read(stream, FRAME.size - 8)
        _, step, time, cells, crc = FRAME.unpack(raw)
        if (cells != header.cells or not np.isfinite(time) or time != step * header.dt or
                (not count and step != header.start_step) or
                (count and (step <= last or step > next_sample(last, header))) or
                (count > 1 and not on_grid(last, header))):
            raise ValueError('Invalid accepted-step clock or sampling grid.')
        data = _read(stream, cells * ROW.itemsize)
        if zlib.crc32(data, zlib.crc32(raw[:28] + bytes(4))) != crc:
            raise ValueError('Relaxation frame checksum mismatch.')
        rows = np.frombuffer(data, dtype=ROW)
        ids, values = rows['id'], rows['values']
        if (np.any(ids < 0) or np.any(ids[1:] <= ids[:-1]) or
                (identity is not None and not np.array_equal(ids, identity)) or
                not np.all(np.isfinite(values)) or np.any(values[:, 3:6] <= 0)):
            raise ValueError('Invalid cell identity or spatial observation.')
        identity = ids
        count += 1
        last = step
        yield Snapshot(step, time, ids, values)


def increment(first, last, dt):
    """Integrate rates, not a fitted displacement residual; exact at stride one.

    For S accepted steps, interpolate the *pre-step* rates at n=0..S-1.
    The shape-dependent advection factor is left-held: the velocity within
    the interval is available only through its integral. Test convergence.
    """
    steps = last.step - first.step
    if steps <= 0 or not np.array_equal(first.ids, last.ids):
        raise ValueError('Intervals require increasing steps and identical cell IDs.')
    a, b = first.values, last.values
    result = {'displacement': b[:, CENTROID] - a[:, CENTROID]}
    for name, columns in (('propulsion', PROPULSION), ('advection', ADVECTION)):
        result[name] = np.zeros_like(result['displacement'])
        result[name][:, :2] = b[:, columns] - a[:, columns]
    result['interaction'] = result['advection'] - result['propulsion']
    for name, columns in (('interfacial', INTERFACIAL), ('overlap', OVERLAP)):
        result[name] = dt * (steps * a[:, columns] +
                            (steps - 1) * .5 * (b[:, columns] - a[:, columns]))
    result['advection_correction'] = np.zeros_like(result['displacement'])
    result['advection_correction'][:, :2] = (a[:, ADVECTION_FACTOR] - 1) * result['advection'][:, :2]
    result['residual'] = result['displacement'] - (result['advection'] +
        result['interfacial'] + result['overlap'] + result['advection_correction'])
    return result


def integrate_segment(path):
    """O(N) memory. A corrupt or incomplete segment never returns a result."""
    with Path(path).open('rb') as stream:
        header = read_header(stream)
        previous, first, totals = None, None, None
        for current in snapshots(stream, header):
            if previous is not None:
                delta = increment(previous, current, header.dt)
                if totals is None:
                    totals = delta
                else:
                    for name in totals:
                        totals[name] += delta[name]
            else:
                first = current
            previous = current
        if totals is None:
            raise ValueError('At least two snapshots are required for an interval.')
        return header, first, previous, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('segment', type=Path)
    args = parser.parse_args()
    header, first, last, totals = integrate_segment(args.segment)
    print(json.dumps(dict(start_step=first.step, end_step=last.step,
        duration=(last.step-first.step)*header.dt, interval=header.interval,
        metadata=header.metadata, cells=[dict(id=int(gid),
            **{name: values[i].tolist() for name, values in totals.items()})
            for i, gid in enumerate(first.ids)]), indent=2))


if __name__ == '__main__':
    main()
