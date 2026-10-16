#!/usr/bin/env python3
"""Read committed PF3D interface frames and recover exact crossing-cube samples.

This is geometry, not a restart or a contact/T-event definition. Only the
phi=0.5 interface is recoverable. See docs/surface_output_3d.md.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import zlib

FILE = struct.Struct('<8s8I3q6dQ2q3d4Q')
FRAME = struct.Struct('<8sQd3Q')
CELL = struct.Struct('<4q2IQ10f2I')
CHUNK = struct.Struct('<8s3Q')
FOOTER = struct.Struct('<8s3Q')
VERTEX = struct.Struct('<Qf')
MAX_CHUNK_BYTES = (64 * 1024 * 1024 // VERTEX.size) * VERTEX.size


class SurfaceFormatError(ValueError):
    pass


class IncompleteSurfaceStream(SurfaceFormatError):
    pass


@dataclass(frozen=True)
class Header:
    cells: int
    boundary_flags: int
    codec: int
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    dt: float
    tau: float
    level: float
    interval: int
    channel_height: int
    channel_padding: int
    wall_kappa: float
    wall_width: float
    interface_width: float

    @property
    def geometry(self):
        return {3: 'slab', 7: 'periodic', 11: 'channel'}[self.boundary_flags]


@dataclass(frozen=True)
class Cell:
    id: int
    origin: tuple[int, int, int]
    edge: int
    gamma: float
    active_speed: float
    radius: float
    polarity: tuple[float, float, float]
    velocity: tuple[float, float, float]
    tumbles: int
    flags: int
    data: bytes

    @property
    def vertex_count(self):
        return len(self.data) // VERTEX.size

    def samples(self):
        """Yield (padded-grid index, original float32 value), without NumPy."""
        return VERTEX.iter_unpack(self.data)

    def arrays(self):
        """Sorted index/value arrays. NumPy is required only for analysis."""
        import numpy as np
        records = np.frombuffer(self.data, dtype=[('index', '<u8'), ('phi', '<f4')])
        order = np.argsort(records['index'])
        indices, values = records['index'][order], records['phi'][order]
        side = self.edge + 2
        if (len(indices) and (indices[-1] >= side**3 or
                np.any(indices[1:] == indices[:-1]))) or not np.all(np.isfinite(values)):
            raise SurfaceFormatError('invalid, duplicate, or non-finite interface vertex')
        return indices, values


@dataclass(frozen=True)
class Frame:
    step: int
    time: float
    cells: tuple[Cell, ...]
    stored_bytes: int

    @property
    def vertex_count(self):
        return sum(c.vertex_count for c in self.cells)


class SurfaceReader:
    """One bounded frame in memory; only committed frames are yielded.

    Read to exhaustion to check the completion marker. recover=True permits a
    truncated tail, not corruption. Frame/cell byte limits are explicit, not
    allocations inferred unconditionally from an untrusted file.
    """
    def __init__(self, path, *, recover=False, max_frame_bytes=512 * 1024 * 1024):
        if not isinstance(max_frame_bytes, int) or max_frame_bytes <= 0:
            raise ValueError('max_frame_bytes must be positive')
        self.file = Path(path).open('rb')
        self.recover = recover
        self.max_frame_bytes = max_frame_bytes
        self.complete = False
        self.truncated = False
        self.frame_count = 0
        self.last_step = None
        self.identity = None
        self.frame_crc = 0
        try:
            raw = self._read(FILE.size)
            values = FILE.unpack(raw)
            (magic, version, header_bytes, cells, flags, codec, vertex_bytes,
             cell_bytes, frame_bytes, nx, ny, nz, dx, dy, dz, dt, tau, level,
             interval, height, padding, wall_kappa, wall_width, width,
             checksum, classification, shell, reserved) = values
            canonical = bytearray(raw)
            struct.pack_into('<Q', canonical, 160, 0)
            if (magic != b'PF3SUR1\0' or version != 1 or header_bytes != FILE.size
                    or vertex_bytes != VERTEX.size or cell_bytes != CELL.size
                    or frame_bytes != FRAME.size or flags not in (3, 7, 11)
                    or codec not in (0, 1) or not 0 < cells <= 1_000_000
                    or min(nx, ny, nz) <= 0 or nx != ny or (flags == 7 and nx != nz)
                    or interval <= 0 or level != 0.5 or classification != 1 or shell != 1
                    or reserved != 0 or zlib.crc32(canonical) != checksum
                    or any(not math.isfinite(v) or v <= 0 for v in (dx, dy, dz, dt, tau, width))):
                raise SurfaceFormatError('invalid or unsupported surface header')
            if flags == 11:
                if (height <= 0 or padding <= 0 or nz != height + 2 * padding
                        or not math.isfinite(wall_kappa) or wall_kappa <= 0
                        or not math.isfinite(wall_width) or wall_width <= 0):
                    raise SurfaceFormatError('invalid resolved channel geometry')
            elif (height, padding, wall_kappa, wall_width) != (0, 0, 0.0, 0.0):
                raise SurfaceFormatError('non-channel file carries channel walls')
            self.header = Header(cells, flags, codec, nx, ny, nz, dx, dy, dz,
                                 dt, tau, level, interval, height, padding,
                                 wall_kappa, wall_width, width)
        except Exception:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()

    def _read(self, size, *, frame=False):
        data = self.file.read(size)
        if len(data) != size:
            raise IncompleteSurfaceStream('surface stream has an incomplete tail')
        if frame:
            self.frame_crc = zlib.crc32(data, self.frame_crc)
        return data

    def _chunk(self, remaining):
        raw_header = self._read(CHUNK.size, frame=True)
        magic, count, stored, checksum = CHUNK.unpack(raw_header)
        decoded_bytes = count * VERTEX.size
        if (magic != b'P3SCHK1\0' or not 0 < count <= remaining
                or decoded_bytes > MAX_CHUNK_BYTES or stored <= 0
                or stored > MAX_CHUNK_BYTES + 1024 * 1024):
            raise SurfaceFormatError('invalid or over-limit surface chunk')
        data = self._read(stored, frame=True)
        if self.header.codec == 1:
            try:
                import zstandard
            except ImportError as error:
                raise SurfaceFormatError('install zstandard to read compressed surfaces') from error
            try:
                data = zstandard.ZstdDecompressor().decompress(data, max_output_size=decoded_bytes)
            except zstandard.ZstdError as error:
                raise SurfaceFormatError('invalid zstd surface chunk') from error
        if len(data) != decoded_bytes or zlib.crc32(data) != checksum:
            raise SurfaceFormatError('surface chunk size or checksum mismatch')
        return data, stored

    def _next_frame(self):
        magic = self._read(8)
        if magic == b'P3SEND1\0':
            _, frames, step, reserved = FOOTER.unpack(magic + self._read(FOOTER.size - 8))
            if (frames != self.frame_count or step != (self.last_step or 0)
                    or reserved != 0 or self.file.read(1)):
                raise SurfaceFormatError('invalid surface completion marker')
            self.complete = True
            return None
        raw = magic + self._read(FRAME.size - 8)
        magic, step, time, cells, vertices, reserved = FRAME.unpack(raw)
        if (magic != b'P3SFRM1\0' or cells != self.header.cells or reserved
                or (self.last_step is not None and step <= self.last_step)
                or not math.isfinite(time) or time != step * self.header.dt
                or cells * CELL.size + vertices * VERTEX.size > self.max_frame_bytes):
            raise SurfaceFormatError('invalid or over-limit surface frame')
        self.frame_crc = zlib.crc32(raw)
        result = []
        total = stored_total = 0
        for _ in range(cells):
            raw_cell = self._read(CELL.size, frame=True)
            (gid, ox, oy, oz, edge, cell_reserved, count, gamma, speed, radius,
             reserved_float, px, py, pz, vx, vy, vz, tumbles, flags) = CELL.unpack(raw_cell)
            if (gid < 0 or not 0 < edge <= 4096 or edge % 8 or cell_reserved or reserved_float
                    or count > (edge + 2)**3 or total + count > vertices
                    or any(not math.isfinite(v) for v in (gamma, speed, radius, px, py, pz, vx, vy, vz))
                    or gamma <= 0 or speed < 0 or radius <= 0
                    or min(ox, oy, oz) <= -(1 << 63)
                    or max(ox, oy, oz) > (1 << 63) - 1 - edge):
                raise SurfaceFormatError('invalid surface cell metadata')
            chunks = []
            remaining = count
            while remaining:
                data, stored = self._chunk(remaining)
                chunks.append(data)
                remaining -= len(data) // VERTEX.size
                stored_total += stored
            result.append(Cell(gid, (ox, oy, oz), edge, gamma, speed, radius,
                               (px, py, pz), (vx, vy, vz), tumbles, flags, b''.join(chunks)))
            total += count
        footer_magic, footer_step, footer_vertices, checksum = FOOTER.unpack(self._read(FOOTER.size))
        ids = tuple(sorted(c.id for c in result))
        if (footer_magic != b'P3SCMT1\0' or footer_step != step or footer_vertices != vertices
                or total != vertices or checksum != self.frame_crc or len(set(ids)) != cells
                or (self.identity is not None and ids != self.identity)):
            raise SurfaceFormatError('invalid frame commit, checksum, or persistent identities')
        self.identity = ids
        self.last_step = step
        self.frame_count += 1
        return Frame(step, time, tuple(result), stored_total)

    def __iter__(self):
        while not self.complete:
            try:
                frame = self._next_frame()
            except IncompleteSurfaceStream:
                self.truncated = True
                if not self.recover:
                    raise
                return
            if frame is None:
                return
            yield frame


def crossing_cubes(cell):
    """Return padded lower corners and exact eight samples of crossing cubes.

    Corner order is (000,100,010,110,001,101,011,111). Missing interior samples
    are NOT zero: only fully present mixed-sign cubes are retained. Filling the
    missing cell interior with zero would create a spurious internal surface.
    """
    import numpy as np
    indices, values = cell.arrays()
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 8), dtype=np.float32)
    side = cell.edge + 2
    x, y, z = indices % side, (indices // side) % side, indices // (side * side)
    candidates = []
    offsets = []
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                offset = (dz * side + dy) * side + dx
                offsets.append(offset)
                keep = ((x >= dx) & (x < side - 1 + dx) & (y >= dy)
                        & (y < side - 1 + dy) & (z >= dz) & (z < side - 1 + dz))
                candidates.append(indices[keep] - np.uint64(offset))
    lower = np.unique(np.concatenate(candidates))
    wanted = lower[:, None] + np.asarray(offsets, dtype=np.uint64)[None, :]
    locations = np.searchsorted(indices, wanted)
    safe = np.minimum(locations, len(indices) - 1)
    present = np.all((locations < len(indices)) & (indices[safe] == wanted), axis=1)
    lower, safe = lower[present], safe[present]
    phi = values[safe]
    inside = phi >= np.float32(0.5)
    mixed = np.any(inside, axis=1) & ~np.all(inside, axis=1)
    lower, phi = lower[mixed], phi[mixed]
    coordinates = np.column_stack((lower % side, (lower // side) % side, lower // (side * side)))
    return coordinates.astype(np.int32), phi


def world_cube_origins(cell, coordinates):
    """Unwrapped lattice coordinates, not periodic-image-clipped positions."""
    import numpy as np
    return coordinates.astype(np.int64) + np.asarray(cell.origin, dtype=np.int64) - 1


def closure_flags(header, cell, coordinates):
    """Bit1: support-shell closure; bit2: bounded-domain exterior closure.

    These artificial caps must not be classified as free cell-cell contacts.
    Resolved steric walls inside the computational array are not zero caps.
    """
    import numpy as np
    flags = np.any((coordinates == 0) | (coordinates == cell.edge), axis=1).astype(np.uint8)
    if header.boundary_flags != 7:
        z = world_cube_origins(cell, coordinates)[:, 2]
        flags |= (((z < 0) | (z + 1 >= header.nz)).astype(np.uint8) << 1)
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--max-frame-mib', type=int, default=512)
    args = parser.parse_args()
    with SurfaceReader(args.path, recover=args.recover,
                       max_frame_bytes=args.max_frame_mib * 1024 * 1024) as reader:
        print(json.dumps({'geometry': reader.header.geometry, 'cells': reader.header.cells,
                          'interval_steps': reader.header.interval, 'level': reader.header.level}))
        for frame in reader:
            print(json.dumps({'step': frame.step, 'time_tau': frame.time / reader.header.tau,
                              'vertices': frame.vertex_count, 'stored_bytes': frame.stored_bytes}))
        print(json.dumps({'complete': reader.complete, 'committed_frames': reader.frame_count,
                          'truncated': reader.truncated}))


if __name__ == '__main__':
    main()
