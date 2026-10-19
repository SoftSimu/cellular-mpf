#!/usr/bin/env python3
"""Export selected recorded interfaces as standard binary PLY meshes and metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from pf3d_contact import UnobservableInterface, mesh_from_surface
from pf3d_surface import SurfaceReader, SurfaceFormatError, frame_digest, geometry_contract


def write_ply(path, mesh, shift):
    vertices = np.asarray(mesh.vertices + shift, dtype='<f8')
    if len(vertices) >= 2**31 or not np.all(np.isfinite(vertices)):
        raise ValueError('mesh vertices exceed PLY index or coordinate limits')
    faces = np.empty(len(mesh.faces), dtype=[('count', 'u1'), ('vertices', '<i4', (3,))])
    faces['count'], faces['vertices'] = 3, mesh.faces
    header = (f'ply\nformat binary_little_endian 1.0\ncomment global_cell_id {mesh.id}\n'
              f'element vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\n'
              f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n').encode('ascii')
    digest = hashlib.sha256()
    with path.open('xb') as output:
        for data in (header, vertices.tobytes(), faces.tobytes()):
            output.write(data); digest.update(data)
    return digest.hexdigest()


def export(args):
    if (args.max_frame_mib <= 0 or args.max_cell_voxels <= 0
            or any(step < 0 for step in args.steps)
            or args.ids is not None and any(gid < 0 for gid in args.ids)):
        raise ValueError('steps/IDs must be nonnegative and memory limits positive')
    requested_steps = set(args.steps)
    requested_ids = None if args.ids is None else set(args.ids)
    if args.anchor_id is not None and (args.anchor_id < 0
            or requested_ids is not None and args.anchor_id not in requested_ids):
        raise ValueError('anchor ID must be one of the selected cells')
    args.output_dir.mkdir()
    manifest = dict(state='RUNNING', format='binary_little_endian_PLY', frames=[], streams=[],
        surface_level=.5, interpretation='interpolated real interfaces; artificial computational caps removed')
    last_step = last_digest = identity = contract = None
    try:
        for path in args.inputs:
            with SurfaceReader(path, max_frame_bytes=args.max_frame_mib * 1024 * 1024) as reader:
                current = geometry_contract(reader.header)
                if contract is not None and current != contract:
                    raise SurfaceFormatError('surface legs have different physical/geometry contracts')
                contract = current
                last_frame = None
                for frame in reader:
                    ids = tuple(sorted(cell.id for cell in frame.cells))
                    if identity is not None and ids != identity:
                        raise SurfaceFormatError('persistent cell IDs change between surface legs')
                    identity = ids
                    if frame.step == last_step:
                        if frame_digest(frame) != last_digest:
                            raise SurfaceFormatError('restart endpoint differs in metadata or exact samples')
                        last_frame = frame
                        continue
                    if last_step is not None and frame.step < last_step:
                        raise SurfaceFormatError('surface legs overlap or are not in increasing step order')
                    last_step, last_frame = frame.step, frame
                    if frame.step not in requested_steps:
                        continue
                    wanted = set(ids) if requested_ids is None else requested_ids
                    if not wanted <= set(ids):
                        raise ValueError('selected cell ID not present in the recording')
                    entries, meshes = [], {}
                    for cell in frame.cells:
                        if cell.id not in wanted:
                            continue
                        entry = dict(id=cell.id, origin=cell.origin, brick_edge=cell.edge,
                            gamma=cell.gamma, radius=cell.radius, active_speed=cell.active_speed,
                            polarity=cell.polarity, velocity=cell.velocity, tumble_count=cell.tumbles,
                            recorded_samples=cell.vertex_count, recorder_flags=cell.flags)
                        try:
                            mesh = mesh_from_surface(cell, reader.header,
                                                     max_voxels=args.max_cell_voxels)
                            if not len(mesh.faces):
                                raise UnobservableInterface('empty or zero-area 0.5 interface')
                            meshes[cell.id] = mesh
                            entry['observable'] = True
                        except UnobservableInterface as error:
                            entry.update(observable=False, reason=str(error))
                        entries.append(entry)
                    anchor = None
                    if args.anchor_id is not None:
                        if args.anchor_id not in meshes:
                            raise ValueError('the anchor cell has no observable interface')
                        anchor = meshes[args.anchor_id].vertices.mean(axis=0)
                    lengths = np.array([reader.header.nx * reader.header.dx,
                                        reader.header.ny * reader.header.dy,
                                        reader.header.nz * reader.header.dz])
                    periodic = 3 if reader.header.boundary_flags == 7 else 2
                    for entry in entries:
                        if not entry['observable']:
                            continue
                        mesh = meshes[entry['id']]
                        shift = np.zeros(3)
                        if anchor is not None:
                            centre = mesh.vertices.mean(axis=0)
                            shift[:periodic] = np.rint((anchor[:periodic] - centre[:periodic])
                                                      / lengths[:periodic]) * lengths[:periodic]
                        name = f'step-{frame.step:012d}-cell-{entry["id"]}.ply'
                        digest = write_ply(args.output_dir / name, mesh, shift)
                        entry.update(file=name, sha256=digest, image_shift=shift.tolist(),
                                     vertices=len(mesh.vertices), triangles=len(mesh.faces),
                                     removed_cap_area=mesh.removed_cap_area)
                    manifest['frames'].append(dict(step=frame.step, time_tau=frame.time / reader.header.tau,
                        source=str(path), cells=sorted(entries, key=lambda item: item['id'])))
                if not reader.complete:
                    raise SurfaceFormatError('surface stream did not finish cleanly')
                manifest['streams'].append(dict(path=str(path), complete=True, frames=reader.frame_count))
                if last_frame is not None:
                    last_digest = frame_digest(last_frame)
        found = {frame['step'] for frame in manifest['frames']}
        if found != requested_steps:
            raise ValueError('requested steps are absent: ' + str(sorted(requested_steps - found)))
        manifest.update(state='PASS', geometry=contract,
                        time_reference='absolute simulator time in tau; includes any preparation')
    except Exception as error:
        manifest.update(state='FAIL', error=str(error))
        raise
    finally:
        with (args.output_dir / 'manifest.json').open('x', encoding='utf-8') as output:
            json.dump(manifest, output, indent=2, allow_nan=False)
            output.write('\n')
    return dict(state='PASS', frames=len(manifest['frames']), output=str(args.output_dir))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+', help='ordered complete surface legs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', required=True, help='absolute accepted steps to export')
    parser.add_argument('--ids', type=int, nargs='+', help='persistent cell IDs; omitted means all cells')
    parser.add_argument('--anchor-id', type=int, help='place whole selected cells in the closest periodic image to this cell')
    parser.add_argument('--max-frame-mib', type=int, default=512)
    parser.add_argument('--max-cell-voxels', type=int, default=32_000_000)
    try:
        result = export(parser.parse_args(argv))
    except (OSError, ValueError, KeyError, ImportError) as error:
        print('surface export: ' + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
