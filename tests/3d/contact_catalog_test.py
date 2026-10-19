"""Whole-history reconciliation and plain-table export of graph candidates."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import pf3d_analyze_contacts as analysis
from pf3d_event_catalog import collect_catalog
from pf3d_exchange import ContactTracker


def history(observations, ids=(1, 2, 3, 4)):
    options = dict(area_on=10., area_off=5., persistence_steps=10,
                   max_frame_gap=20, exchange_window=30)
    tracker = ContactTracker(**options)
    rows = [dict(type='metadata', schema='pf3d-exchanges-1', **options,
                 contact_metadata=dict(geometry=dict(dt=.01, tau=10000., cells=len(ids))))]
    counts = dict(contact_gained=0, contact_lost=0, four_cell_candidates=0,
                  three_dimensional_graph_candidates=0, observation_resets=0,
                  internally_censored_changes=0, internally_censored_establishments=0)
    for step, edges in observations:
        update = tracker.update(step, ids, {tuple(sorted(e)): (100., 100.) for e in edges})
        rows.append(dict(type='frame', time_tau=step * .01 / 10000., **asdict(update)))
        for change in update.changes:
            counts[change.kind] += 1
        counts['four_cell_candidates'] += len(update.exchange_candidates)
        counts['three_dimensional_graph_candidates'] += len(update.three_dimensional_graph_candidates)
        counts['observation_resets'] += update.observation_reset
        for pending in update.censored_pending:
            key = 'internally_censored_establishments' if pending['from_state'] is None else 'internally_censored_changes'
            counts[key] += 1
    rows.append(dict(type='summary', source_status='complete', frames=len(observations), counts=counts,
        right_censored_pending=[p.record(pair, 'end_of_stream') for pair, p in tracker.pending.items()],
        confirmed_3d_topological_events=None))
    return json.loads(json.dumps(rows))


FLANKS = [(1, 3), (1, 4), (2, 3), (2, 4)]


class EventCatalog(unittest.TestCase):
    def swap(self):
        return history([(0, FLANKS + [(1, 2)]), (10, FLANKS + [(3, 4)]),
                        (20, FLANKS + [(3, 4)])])

    def test_unambiguous_candidate_and_time_reference(self):
        result = collect_catalog(self.swap())
        self.assertEqual(result['counts']['unambiguous_four_cell_candidates'], 1)
        self.assertEqual(result['counts']['contact_gained'], 1)
        self.assertEqual(result['counts']['contact_lost'], 1)
        candidate, = result['exchanges']
        self.assertFalse(candidate['ambiguous'])
        self.assertEqual(candidate['interval'], [0, 10])
        self.assertEqual(candidate['interval_tau'], [0., .00001])
        self.assertIsNone(result['confirmed_3d_topological_events'])

    def test_later_alternative_reconciles_both_pairings(self):
        flanks = [(a, b) for a in (1, 2) for b in (3, 4, 5, 6)]
        rows = history([(0, flanks + [(1, 2)]), (10, flanks + [(3, 4)]),
                        (20, flanks + [(3, 4)]), (30, flanks + [(3, 4), (5, 6)]),
                        (40, flanks + [(3, 4), (5, 6)])], tuple(range(1, 7)))
        result = collect_catalog(rows)
        self.assertEqual(result['counts']['ambiguous_four_cell_candidates'], 2)
        self.assertEqual([c['conflicting_candidates'] for c in result['exchanges']], [[1], [0]])

    def test_repeated_cells_at_separate_times_are_not_competing(self):
        rows = history([(0, FLANKS + [(1, 2)]), (10, FLANKS + [(3, 4)]),
                        (20, FLANKS + [(3, 4)]), (30, FLANKS + [(1, 2)]),
                        (40, FLANKS + [(1, 2)])])
        result = collect_catalog(rows)
        self.assertEqual(len(result['exchanges']), 2)
        self.assertTrue(all(not c['ambiguous'] for c in result['exchanges']))

    def test_bulk_graph_annotation_is_not_a_confirmed_topological_event(self):
        ids = tuple(range(1, 6))
        all_edges = [(a, b) for a in ids for b in ids if a < b]
        before = [p for p in all_edges if p != (1, 2)]
        rows = history([(0, before), (10, all_edges), (20, all_edges)], ids)
        candidate, = collect_catalog(rows)['bulk_candidates']
        self.assertEqual(candidate['kind'], 'ET_compatible_graph_candidate')
        self.assertFalse(candidate['confirmed_topological_transition'])

    def test_pending_change_stays_censored(self):
        rows = history([(0, []), (10, [(1, 2)])], (1, 2))
        result = collect_catalog(rows)
        self.assertFalse(result['changes'])
        self.assertEqual(result['counts']['right_censored_pending'], 1)

    def test_incomplete_invalid_or_modified_table_is_rejected(self):
        rows = self.swap()
        with self.assertRaises(ValueError): collect_catalog(rows[:-1])
        with self.assertRaises(ValueError): collect_catalog(rows + [rows[-1]])
        for field in ('contact_gained', 'four_cell_candidates'):
            modified = copy.deepcopy(rows)
            modified[-1]['counts'][field] += 1
            with self.assertRaises(ValueError): collect_catalog(modified)
        modified = copy.deepcopy(rows)
        modified[-2]['changes'][0]['index'] += 1
        with self.assertRaises(ValueError): collect_catalog(modified)
        modified = copy.deepcopy(rows)
        modified[-2]['exchange_candidates'][0]['confirmed_topological_T1'] = True
        with self.assertRaises(ValueError): collect_catalog(modified)

    def test_prefix_requires_explicit_choice(self):
        rows = self.swap(); rows[-1]['source_status'] = 'recovered_prefix'
        with self.assertRaises(ValueError): collect_catalog(rows)
        self.assertEqual(collect_catalog(rows, allow_prefix=True)['source_status'], 'recovered_prefix')

    def test_csv_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='pf3d-catalog-') as tmp:
            root = Path(tmp); source = root / 'changes.jsonl'; target = root / 'catalog'
            source.write_text(''.join(json.dumps(row) + '\n' for row in self.swap()))
            args = ['catalog', str(source), '--output-dir', str(target)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(analysis.main(args), 0)
                self.assertEqual(analysis.main(args), 2)
            import csv
            with (target / 'contact_changes.csv').open(newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            result = json.loads((target / 'catalog.json').read_text())
            self.assertEqual(result['counts']['unambiguous_four_cell_candidates'], 1)
            self.assertFalse(result['exchanges'][0]['confirmed_topological_T1'])


if __name__ == '__main__':
    unittest.main()
