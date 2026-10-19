"""Finalize graph-change tables without treating provisional motifs as T events."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('expected two persistent cell IDs')
    a, b = (integer(v, 'cell ID') for v in value)
    if a >= b:
        raise ValueError('cell pairs must be distinct and in increasing ID order')
    return a, b


def collect_catalog(records, *, allow_prefix=False):
    records = iter(records)
    metadata = next(records, {})
    if metadata.get('type') != 'metadata' or metadata.get('schema') != 'pf3d-exchanges-1':
        raise ValueError('not a supported exchange table')
    geometry = metadata['contact_metadata']['geometry']
    dt, tau = geometry['dt'], geometry['tau']
    if not all(math.isfinite(x) and x > 0 for x in (dt, tau)):
        raise ValueError('invalid simulation time units')
    changes, candidates, bulk = [], [], []
    keys = set()
    summary = last_step = None
    frames = resets = censored_changes = censored_establishments = 0
    first_step = last_pending = None
    for record in records:
        if summary is not None:
            raise ValueError('records follow exchange completion summary')
        if record.get('type') == 'summary':
            summary = record
            continue
        if record.get('type') != 'frame':
            raise ValueError('failed or unsupported exchange-table record')
        step = integer(record['step'], 'accepted step')
        if last_step is not None and step <= last_step:
            raise ValueError('exchange-table steps must strictly increase')
        if not math.isclose(record['time_tau'], step * dt / tau, rel_tol=0, abs_tol=1e-12):
            raise ValueError('exchange-table time does not match the accepted step')
        first_step = step if first_step is None else first_step
        last_step, frames = step, frames + 1
        resets += int(record['observation_reset'])
        last_pending = integer(record['pending_count'], 'pending count')
        for pending in record['censored_pending']:
            if pending['from_state'] is None:
                censored_establishments += 1
            else:
                censored_changes += 1
        for item in record['changes']:
            index = integer(item['index'], 'change index')
            if index != len(changes) or item['kind'] not in ('contact_gained', 'contact_lost'):
                raise ValueError('missing, duplicated or invalid contact change')
            a, b = pair(item['pair'])
            previous = integer(item['last_previous_step'], 'previous step')
            changed = integer(item['first_changed_step'], 'first changed step')
            if not previous < changed <= item['confirmed_step'] == step:
                raise ValueError('invalid change observation/confirmation interval')
            changes.append(dict(change_id=index, kind=item['kind'], cell_a=a, cell_b=b,
                previous_step=previous, first_changed_step=changed, confirmed_step=step,
                previous_tau=previous * dt / tau, first_changed_tau=changed * dt / tau,
                confirmed_tau=step * dt / tau))
        for raw in record['exchange_candidates']:
            lost, gained = raw['lost_change'], raw['gained_change']
            integer(lost, 'lost change'); integer(gained, 'gained change')
            if max(lost, gained) >= len(changes) or (lost, gained) in keys:
                raise ValueError('unknown change or duplicate exchange candidate')
            before, after = changes[lost], changes[gained]
            a = pair(raw['lost_pair']); b = pair(raw['gained_pair'])
            if (before['kind'] != 'contact_lost' or after['kind'] != 'contact_gained'
                    or a != (before['cell_a'], before['cell_b'])
                    or b != (after['cell_a'], after['cell_b']) or len(set(a + b)) != 4):
                raise ValueError('exchange candidate does not match its contact changes')
            interval = [min(before['previous_step'], after['previous_step']),
                        max(before['first_changed_step'], after['first_changed_step'])]
            if (raw['interval'] != interval or raw['confirmed_step'] != step
                    or raw.get('confirmed_topological_T1') is not False):
                raise ValueError('invalid exchange interval or unsupported topological claim')
            flanks = sorted((min(x, y), max(x, y)) for x in a for y in b)
            if sorted(pair(p) for p in raw['stable_flanks']) != flanks:
                raise ValueError('exchange candidate does not have the four cross-links')
            keys.add((lost, gained))
            candidates.append(dict(candidate_id=len(candidates), lost_change=lost,
                gained_change=gained, lost_pair=a, gained_pair=b, cells=sorted(a + b),
                interval=interval, confirmed_step=step,
                interval_tau=[v * dt / tau for v in interval],
                tracker_ambiguous=bool(raw['ambiguous_pairing']),
                prior_pairing_conflicts=raw['prior_pairing_conflicts'],
                confirmed_topological_T1=False))
        for raw in record['three_dimensional_graph_candidates']:
            change_id = integer(raw['change'], 'bulk change index')
            if change_id >= len(changes):
                raise ValueError('bulk candidate references an unknown change')
            expected = ('ET_compatible_graph_candidate' if changes[change_id]['kind'] == 'contact_gained'
                        else 'TE_compatible_graph_candidate')
            if (raw['kind'] != expected or raw.get('confirmed_topological_transition') is not False
                    or pair(raw['pair']) != (changes[change_id]['cell_a'], changes[change_id]['cell_b'])):
                raise ValueError('invalid bulk graph candidate')
            bulk.append(dict(candidate_id=len(bulk), **raw))
    counts = Counter(item['kind'] for item in changes)
    expected_counts = dict(contact_gained=counts['contact_gained'], contact_lost=counts['contact_lost'],
        four_cell_candidates=len(candidates), three_dimensional_graph_candidates=len(bulk),
        observation_resets=resets, internally_censored_changes=censored_changes,
        internally_censored_establishments=censored_establishments)
    if (summary is None or not frames or summary.get('frames') != frames
            or summary.get('counts') != expected_counts
            or len(summary['right_censored_pending']) != last_pending
            or summary.get('confirmed_3d_topological_events') is not None):
        raise ValueError('missing or inconsistent exchange completion summary')
    status = summary.get('source_status')
    if status not in ('complete', 'bounded_prefix', 'recovered_prefix'):
        raise ValueError('unknown source completion status')
    if status != 'complete' and not allow_prefix:
        raise ValueError('prefix observations require explicit --allow-prefix')

    # Reconcile the whole observed history: a later competing pairing can
    # invalidate an earlier provisional assignment, not just the newer one.
    by_change, by_cell = defaultdict(set), defaultdict(list)
    by_key = {(c['lost_change'], c['gained_change']): c['candidate_id'] for c in candidates}
    conflicts = [set() for _ in candidates]
    for c in candidates:
        cid = c['candidate_id']
        for index in (c['lost_change'], c['gained_change']):
            by_change[index].add(cid)
        for gid in c['cells']:
            by_cell[gid].append(cid)
        for prior in c['prior_pairing_conflicts']:
            if tuple(prior) not in by_key:
                raise ValueError('candidate conflict references an absent pairing')
            other = by_key[tuple(prior)]
            conflicts[cid].add(other); conflicts[other].add(cid)
    for uses in by_change.values():
        for cid in uses:
            conflicts[cid].update(uses - {cid})
    for ids in by_cell.values():
        active = []
        for cid in sorted(ids, key=lambda i: candidates[i]['interval'][0]):
            start = candidates[cid]['interval'][0]
            active = [i for i in active if candidates[i]['interval'][1] >= start]
            for other in active:
                conflicts[cid].add(other); conflicts[other].add(cid)
            active.append(cid)
    for c in candidates:
        c['conflicting_candidates'] = sorted(conflicts[c['candidate_id']])
        c['ambiguous'] = c['tracker_ambiguous'] or bool(c['conflicting_candidates'])
        c.pop('prior_pairing_conflicts')
    return dict(metadata=metadata, source_status=status, frames=frames,
        first_step=first_step, last_step=last_step, changes=changes, exchanges=candidates,
        bulk_candidates=bulk, right_censored_pending=summary['right_censored_pending'],
        counts=dict(**expected_counts,
            unambiguous_four_cell_candidates=sum(not c['ambiguous'] for c in candidates),
            ambiguous_four_cell_candidates=sum(c['ambiguous'] for c in candidates),
            right_censored_pending=len(summary['right_censored_pending'])),
        confirmed_3d_topological_events=None,
        time_reference='absolute simulator time in tau; includes any preparation',
        interpretation='measured persistent proximity-graph changes and reconciled graph candidates; no confirmed 3D junction-collapse count')


def _csv(path, rows, fields):
    with path.open('x', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(',', ':'))
                             if isinstance(value, (list, tuple, dict)) else value
                             for key, value in row.items() if key in fields})


def export_catalog(catalog, directory: Path, source: Path):
    directory.mkdir()
    _csv(directory / 'contact_changes.csv', catalog['changes'],
         ['change_id','kind','cell_a','cell_b','previous_step','first_changed_step',
          'confirmed_step','previous_tau','first_changed_tau','confirmed_tau'])
    _csv(directory / 'exchange_candidates.csv', catalog['exchanges'],
         ['candidate_id','lost_change','gained_change','lost_pair','gained_pair','cells',
          'interval','interval_tau','confirmed_step','ambiguous','conflicting_candidates',
          'confirmed_topological_T1'])
    _csv(directory / 'bulk_candidates.csv', catalog['bulk_candidates'],
         ['candidate_id','kind','change','pair','supporting_triads','ambiguous',
          'alternatives_truncated','tricellular_junction_measured','confirmed_topological_transition'])
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    catalog.update(source=str(source), source_sha256=digest.hexdigest())
    with (directory / 'catalog.json').open('x', encoding='utf-8') as output:
        json.dump(catalog, output, allow_nan=False, indent=2)
        output.write('\n')
    return dict(state='PASS', directory=str(directory), counts=catalog['counts'],
                source_status=catalog['source_status'], confirmed_3d_topological_events=None)
