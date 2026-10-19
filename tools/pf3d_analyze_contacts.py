#!/usr/bin/env python3
"""Measure 3D surface proximity, then track persistent graph changes offline.

The two subcommands deliberately separate costly geometry from threshold and
cadence sensitivity tests. Neither modifies simulation data. See
docs/contact_exchange_3d.md for the observation definition and its limitations.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from pf3d_exchange import ContactTracker
from pf3d_surface import SurfaceReader, SurfaceFormatError, frame_digest, geometry_contract


def write_record(output, record):
    output.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
    output.flush()


@contextmanager
def new_output(path, inputs):
    if path.resolve() in {item.resolve() for item in inputs}:
        raise ValueError('output must not be an input file')
    with path.open('x', encoding='utf-8', newline='\n') as output:
        try:
            yield output
        except Exception as error:
            write_record(output, dict(type='failure', message=str(error)))
            raise


def mesh_diagnostics(mesh):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.vstack((mesh.faces[:, (0,1)], mesh.faces[:, (1,2)], mesh.faces[:, (2,0)]))
    graph = coo_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])),
                       shape=(len(mesh.vertices), len(mesh.vertices))).tocsr()
    components = int(connected_components(graph, directed=False, return_labels=False))
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    area = float(mesh.areas.sum())
    return dict(area=area, triangles=len(mesh.faces), components=components,
        boundary_edges=int(np.count_nonzero(counts == 1)),
        nonmanifold_edges=int(np.count_nonzero(counts > 2)),
        half_level_ties=mesh.half_level_ties, removed_cap_area=mesh.removed_cap_area,
        wall_external_area_bounds=[mesh.wall_area_lower, mesh.wall_area_upper],
        extent=(mesh.bounds[1]-mesh.bounds[0]).tolist())


def measure_frame(frame, header, args):
    import numpy as np
    from pf3d_contact import (UnobservableInterface, frame_contacts, image_shifts,
                              mesh_from_surface, proximity_area)

    meshes, diagnostics = [], []
    for cell in sorted(frame.cells, key=lambda item: item.id):
        metadata = dict(id=cell.id, gamma=cell.gamma, radius=cell.radius,
                        active_speed=cell.active_speed)
        try:
            mesh = mesh_from_surface(cell, header, max_voxels=args.max_cell_voxels)
            if not len(mesh.faces):
                raise UnobservableInterface('empty or zero-area 0.5 interface')
            shifts = [s for s in image_shifts(mesh, mesh, header, args.gap) if np.any(s)]
            if shifts and proximity_area(mesh, mesh, shifts, args.gap,
                                          args.resolution, args.max_depth).upper > 0:
                raise UnobservableInterface('interface is within gap of a periodic self-image')
        except UnobservableInterface as error:
            diagnostics.append(dict(**metadata, observable=False,
                                    topology_usable=False, reason=str(error)))
            continue
        detail = mesh_diagnostics(mesh)
        detail.update(**metadata, observable=True,
            topology_usable=(detail['components'] == 1 and detail['nonmanifold_edges'] == 0
                and (detail['boundary_edges'] == 0 or detail['removed_cap_area'] > 0)))
        diagnostics.append(detail)
        meshes.append(mesh)
    contacts = frame_contacts(meshes, header, gap=args.gap, resolution=args.resolution,
                              max_depth=args.max_depth)
    pairs = [dict(a=c.a, b=c.b, a_near_b=asdict(c.a_near_b),
                  b_near_a=asdict(c.b_near_a), lower=c.lower, upper=c.upper) for c in contacts]
    return dict(type='frame', step=frame.step, time_tau=frame.time/header.tau,
                cells=diagnostics, contacts=pairs)


def measure_job(frame, header, parameters):
    options = SimpleNamespace(**parameters)
    start = time.monotonic()
    record = measure_frame(frame, header, options)
    record['analysis_seconds'] = time.monotonic() - start
    return record


def measure(args):
    from pf3d_contact import validate_measurement
    validate_measurement(args.gap, args.resolution, args.max_depth)
    if (min(args.max_frame_mib, args.max_cell_voxels, args.workers) <= 0
            or args.limit_frames is not None and args.limit_frames <= 0):
        raise ValueError('memory and frame limits must be positive')
    headers = []
    for path in args.inputs:
        with SurfaceReader(path) as reader:
            headers.append(reader.header)
    header = headers[0]
    if any(geometry_contract(h) != geometry_contract(header) for h in headers[1:]):
        raise ValueError('surface legs have different physical/geometry contracts')
    count = duplicates = submitted = 0
    last_step = last_digest = first_step = None
    last_queued_step = None
    identity = None
    reports, limited = [], False
    parameters = dict(gap=args.gap, resolution=args.resolution,
                      max_depth=args.max_depth, max_cell_voxels=args.max_cell_voxels)
    with new_output(args.output, args.inputs) as output, ExitStack() as stack:
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=args.workers)) if args.workers > 1 else None
        pending = deque()

        def commit(record, digest):
            nonlocal count, first_step, last_step
            record['surface_sha256'] = digest
            write_record(output, record)
            count += 1
            first_step = record['step'] if first_step is None else first_step
            last_step = record['step']
            print(f"step {last_step}: {len(record['contacts'])} proximity pairs, "
                  f"{record['analysis_seconds']:.2f} CPU s", file=sys.stderr)

        write_record(output, dict(type='metadata', schema='pf3d-contacts-1',
            geometry=geometry_contract(header), gap=args.gap, resolution=args.resolution,
            max_depth=args.max_depth, extractor='lewiner-inclusive-half-level',
            packages={name: importlib.metadata.version(name)
                      for name in ('numpy','scipy','scikit-image')},
            workers=args.workers,
            streams=[dict(path=str(p), interval=h.interval, codec=h.codec)
                     for p,h in zip(args.inputs, headers)],
            definition='symmetric minimum of directed surface areas within gap; not a force or adhesion law'))
        for path in args.inputs:
            with SurfaceReader(path, recover=args.recover,
                    max_frame_bytes=args.max_frame_mib*1024*1024) as reader:
                for frame in reader:
                    ids = tuple(sorted(cell.id for cell in frame.cells))
                    if identity is not None and ids != identity:
                        raise SurfaceFormatError('persistent cell IDs change between surface legs')
                    identity = ids
                    digest = frame_digest(frame)
                    if frame.step == last_queued_step:
                        if digest != last_digest:
                            raise SurfaceFormatError('restart endpoint differs in cell metadata or exact samples')
                        duplicates += 1
                        continue
                    if last_queued_step is not None and frame.step < last_queued_step:
                        raise SurfaceFormatError('surface legs overlap or are not in increasing step order')
                    if pool is None:
                        commit(measure_job(frame, header, parameters), digest)
                    else:
                        pending.append((pool.submit(measure_job, frame, header, parameters), digest))
                        # Executor.map eagerly consumes the stream on older Python
                        # versions. Bound submitted frames instead of queuing the run.
                        if len(pending) >= args.workers:
                            future, identity_digest = pending.popleft()
                            commit(future.result(), identity_digest)
                    submitted += 1
                    last_queued_step, last_digest = frame.step, digest
                    if args.limit_frames is not None and submitted >= args.limit_frames:
                        limited = True
                        break
                reports.append(dict(path=str(path), complete=reader.complete,
                                    truncated=reader.truncated, read_frames=reader.frame_count))
            if limited:
                break
        while pending:
            future, identity_digest = pending.popleft()
            commit(future.result(), identity_digest)
        if not count:
            raise ValueError('no committed surface frames')
        status = 'bounded_prefix' if limited else 'recovered_prefix' if any(r['truncated'] for r in reports) else 'complete'
        result = dict(type='summary', status=status, frames=count,
            deduplicated_endpoints=duplicates, first_step=first_step, last_step=last_step, streams=reports)
        write_record(output, result)
    return result


def read_records(path):
    with path.open(encoding='utf-8') as source:
        while True:
            line = source.readline(64*1024*1024+1)
            if not line:
                return
            if len(line) > 64*1024*1024:
                raise ValueError('JSONL record exceeds 64 MiB')
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('expected a JSON object')
            yield record


def tracking_scores(record):
    cells = record['cells']
    ids = tuple(cell['id'] for cell in cells)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate persistent ID in contact table')
    censored = {cell['id'] for cell in cells if not cell['observable'] or not cell['topology_usable']}
    scores, seen = {}, set()
    for pair in record['contacts']:
        key = tuple(sorted((pair['a'], pair['b'])))
        if key in seen or key[0] == key[1] or not set(key) <= set(ids):
            raise ValueError('invalid or duplicated pair in contact table')
        seen.add(key)
        lower = min(pair['a_near_b']['lower'], pair['b_near_a']['lower'])
        upper = min(pair['a_near_b']['upper'], pair['b_near_a']['upper'])
        if (lower,upper) != (pair['lower'],pair['upper']):
            raise ValueError('inconsistent directed/symmetric area bounds')
        if not censored.intersection(key):
            scores[key] = (lower,upper)
    return ids, scores, censored


def track(args):
    if args.stride < 1:
        raise ValueError('stride must be positive')
    tracker = ContactTracker(area_on=args.area_on, area_off=args.area_off,
        persistence_steps=args.persistence_steps, max_frame_gap=args.max_frame_gap,
        exchange_window=args.exchange_window)
    records = iter(read_records(args.input))
    metadata = next(records, {})
    if metadata.get('type') != 'metadata' or metadata.get('schema') != 'pf3d-contacts-1':
        raise ValueError('not a supported contact table')
    counts = dict(contact_gained=0, contact_lost=0, four_cell_candidates=0,
        three_dimensional_graph_candidates=0, observation_resets=0,
        internally_censored_changes=0, internally_censored_establishments=0)
    selected = seen = 0
    last_step = None
    summary = None
    with new_output(args.output, [args.input]) as output:
        write_record(output, dict(type='metadata', schema='pf3d-exchanges-1',
            contact_metadata=metadata, area_on=args.area_on, area_off=args.area_off,
            persistence_steps=args.persistence_steps, max_frame_gap=args.max_frame_gap,
            exchange_window=args.exchange_window, stride=args.stride,
            interpretation='persistent graph changes and provisional motifs; not verified 3D T transitions'))
        for record in records:
            if summary is not None:
                raise ValueError('records follow contact completion summary')
            if record.get('type') == 'summary':
                summary = record
                continue
            if record.get('type') != 'frame':
                raise ValueError('failed or unsupported contact-table record')
            step = record['step']
            if not isinstance(step,int) or isinstance(step,bool) or step < 0 or last_step is not None and step <= last_step:
                raise ValueError('contact-table steps must strictly increase')
            last_step = step
            ids, scores, censored = tracking_scores(record)
            if len(ids) != metadata['geometry']['cells']:
                raise ValueError('contact-table cell count differs from metadata')
            seen += 1
            if (seen-1) % args.stride:
                continue
            update = tracker.update(step, ids, scores, unobservable_ids=censored)
            selected += 1
            for change in update.changes:
                counts[change.kind] += 1
            counts['four_cell_candidates'] += len(update.exchange_candidates)
            counts['three_dimensional_graph_candidates'] += len(update.three_dimensional_graph_candidates)
            counts['observation_resets'] += int(update.observation_reset)
            for pending in update.censored_pending:
                key = 'internally_censored_establishments' if pending['from_state'] is None else 'internally_censored_changes'
                counts[key] += 1
            write_record(output, dict(type='frame', time_tau=record['time_tau'], **asdict(update)))
        if (summary is None or summary.get('frames') != seen or summary.get('last_step') != last_step
                or summary.get('status') not in ('complete','recovered_prefix','bounded_prefix') or not selected):
            raise ValueError('contact table has no valid completion summary')
        result = dict(type='summary', source_status=summary['status'], frames=selected,
            counts=counts, right_censored_pending=[pending.record(pair,'end_of_stream')
                for pair,pending in sorted(tracker.pending.items())],
            confirmed_3d_topological_events=None)
        write_record(output, result)
    return result


def catalog(args):
    from pf3d_event_catalog import collect_catalog, export_catalog
    result = collect_catalog(read_records(args.input), allow_prefix=args.allow_prefix)
    return export_catalog(result, args.output_dir, args.input)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    measure_parser = commands.add_parser('measure', help='measure committed 3D surface frames')
    measure_parser.add_argument('inputs', type=Path, nargs='+', help='ordered surface legs')
    measure_parser.add_argument('--output', type=Path, required=True)
    measure_parser.add_argument('--gap', type=float, required=True, help='physical distance; requires model calibration')
    measure_parser.add_argument('--resolution', type=float, default=.25, help='physical subtriangle radius')
    measure_parser.add_argument('--max-depth', type=int, default=2)
    measure_parser.add_argument('--max-frame-mib', type=int, default=512)
    measure_parser.add_argument('--max-cell-voxels', type=int, default=32_000_000)
    measure_parser.add_argument('--workers', type=int, default=1,
        help='independent CPU frame workers; at most this many frames are queued')
    measure_parser.add_argument('--recover', action='store_true', help='permit an incomplete tail, not corrupt data')
    measure_parser.add_argument('--limit-frames', type=int, help='diagnostic prefix only; skips terminal stream validation')
    measure_parser.set_defaults(run=measure)
    track_parser = commands.add_parser('track', help='track graph changes without recomputing geometry')
    track_parser.add_argument('input', type=Path)
    track_parser.add_argument('--output', type=Path, required=True)
    track_parser.add_argument('--area-on', type=float, required=True, help='physical area lower bound for formation')
    track_parser.add_argument('--area-off', type=float, required=True, help='physical area upper bound for loss')
    track_parser.add_argument('--persistence-steps', type=int, required=True)
    track_parser.add_argument('--max-frame-gap', type=int, required=True)
    track_parser.add_argument('--exchange-window', type=int, required=True)
    track_parser.add_argument('--stride', type=int, default=1, help='subsample tables for cadence sensitivity')
    track_parser.set_defaults(run=track)
    catalog_parser = commands.add_parser('catalog', help='finalize candidates and export CSV tables')
    catalog_parser.add_argument('input', type=Path)
    catalog_parser.add_argument('--output-dir', type=Path, required=True)
    catalog_parser.add_argument('--allow-prefix', action='store_true',
        help='accept an explicitly recovered/bounded source, still labelled as a prefix')
    catalog_parser.set_defaults(run=catalog)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = args.run(args)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f'contact analysis: {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
