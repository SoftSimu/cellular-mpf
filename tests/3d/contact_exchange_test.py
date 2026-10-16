"""Known graph histories: persistence, censoring and candidate—not T1—labels."""
from itertools import combinations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from pf3d_exchange import ContactTracker


def tracker(persistence=10, max_gap=20, window=20):
    return ContactTracker(area_on=2., area_off=1., persistence_steps=persistence,
                          max_frame_gap=max_gap, exchange_window=window)


def scores(pairs):
    return {tuple(sorted(pair)): (3.,3.) for pair in pairs}


class ExchangeTracking(unittest.TestCase):
    def test_initial_contacts_are_not_births(self):
        t = tracker()
        result = t.update(0,(1,2),scores([(1,2)]))
        self.assertEqual(result.edges,((1,2),))
        self.assertEqual(result.changes,())

    def test_persistence_is_elapsed_steps_not_sample_count(self):
        t = tracker(15)
        t.update(0,(1,2),{})
        self.assertFalse(t.update(10,(1,2),scores([(1,2)])).changes)
        self.assertFalse(t.update(20,(1,2),scores([(1,2)])).changes)
        result = t.update(30,(1,2),scores([(1,2)]))
        change, = result.changes
        self.assertEqual((change.last_previous_step,change.first_changed_step,change.confirmed_step),(0,10,30))

    def test_single_frame_flicker_is_not_an_event(self):
        t = tracker()
        t.update(0,(1,2),{})
        t.update(10,(1,2),scores([(1,2)]))
        result = t.update(20,(1,2),{})
        self.assertEqual(result.edges,())
        self.assertEqual(result.changes,())
        self.assertEqual(result.pending_count,0)

    def test_uncertainty_resets_confirmation(self):
        t = tracker()
        t.update(0,(1,2),{})
        t.update(10,(1,2),scores([(1,2)]))
        result = t.update(20,(1,2),{(1,2):(1.5,2.5)})
        self.assertEqual(result.uncertain_pairs,((1,2),))
        self.assertEqual(result.pending_count,0)
        self.assertFalse(t.update(30,(1,2),scores([(1,2)])).changes)
        self.assertEqual(len(t.update(40,(1,2),scores([(1,2)])).changes),1)

    def test_hysteresis_keeps_existing_contact(self):
        t = tracker()
        t.update(0,(1,2),scores([(1,2)]))
        result = t.update(10,(1,2),{(1,2):(1.4,1.6)})
        self.assertEqual(result.edges,((1,2),))
        t.update(20,(1,2),{})
        self.assertEqual(t.update(30,(1,2),{}).changes[0].kind,'contact_lost')

    def test_zero_off_threshold_handles_exactly_zero_area(self):
        t = ContactTracker(area_on=1.,area_off=0.,persistence_steps=10,
                           max_frame_gap=20,exchange_window=20)
        t.update(0,(1,2),scores([(1,2)]))
        t.update(10,(1,2),{})
        self.assertEqual(t.update(20,(1,2),{}).edges,())

    def test_unobserved_gap_reinitializes_without_fake_changes(self):
        t = tracker(max_gap=15)
        t.update(0,(1,2),scores([(1,2)]))
        result = t.update(100,(1,2),{})
        self.assertTrue(result.observation_reset)
        self.assertFalse(result.changes)

    def test_gap_reports_a_censored_change_separately_from_establishment(self):
        t = tracker(max_gap=15)
        t.update(0,(1,2),{})
        t.update(10,(1,2),scores([(1,2)]))
        result = t.update(100,(1,2),{})
        pending, = result.censored_pending
        self.assertEqual(pending['kind'],'pending_contact_change')
        self.assertFalse(pending['from_state'])
        t.update(110,(1,2),{},unobservable_ids=(1,))
        t.update(120,(1,2),scores([(1,2)]))
        pending, = t.update(130,(1,2),{},unobservable_ids=(1,)).censored_pending
        self.assertEqual(pending['kind'],'pending_state_establishment')
        self.assertIsNone(pending['from_state'])

    def test_ambiguous_initial_state_is_not_a_later_birth(self):
        t = tracker()
        t.update(0,(1,2),{(1,2):(1.5,2.5)})
        t.update(10,(1,2),scores([(1,2)]))
        result = t.update(20,(1,2),scores([(1,2)]))
        self.assertEqual(result.edges,((1,2),))
        self.assertFalse(result.changes)

    def test_missing_cell_censors_existing_contact_and_later_return(self):
        t = tracker()
        t.update(0,(1,2),scores([(1,2)]))
        result = t.update(10,(1,2),{},unobservable_ids=(1,))
        self.assertEqual(result.edges,())
        self.assertEqual(result.unobservable_ids,(1,))
        self.assertFalse(result.changes)
        t.update(20,(1,2),{})
        self.assertFalse(t.update(30,(1,2),{}).changes)
        t.update(40,(1,2),scores([(1,2)]))
        self.assertEqual(t.update(50,(1,2),scores([(1,2)])).changes[0].kind,'contact_gained')

    def test_unseen_pair_when_a_cell_returns_is_not_a_fake_gain(self):
        t = tracker()
        t.update(0,(1,2),{},unobservable_ids=(1,))
        t.update(10,(1,2),scores([(1,2)]))
        result = t.update(20,(1,2),scores([(1,2)]))
        self.assertEqual(result.edges,((1,2),))
        self.assertFalse(result.changes)

    def test_healthy_pairs_continue_while_another_cell_is_unobservable(self):
        t = tracker()
        t.update(0,(1,2,3),{})
        t.update(10,(1,2,3),scores([(1,2)]),unobservable_ids=(3,))
        result = t.update(20,(1,2,3),scores([(1,2)]),unobservable_ids=(3,))
        self.assertEqual(len(result.changes),1)

    def test_four_cell_diagonal_swap_is_a_candidate(self):
        ids = (1,2,3,4)
        flanks = [(1,3),(1,4),(2,3),(2,4)]
        t = tracker()
        t.update(0,ids,scores(flanks+[(1,2)]))
        t.update(10,ids,scores(flanks+[(3,4)]))
        result = t.update(20,ids,scores(flanks+[(3,4)]))
        self.assertEqual(len(result.changes),2)
        item, = result.exchange_candidates
        self.assertEqual(item['lost_pair'],(1,2))
        self.assertEqual(item['gained_pair'],(3,4))
        self.assertFalse(item['confirmed_topological_T1'])
        self.assertTrue(item['pairing_is_provisional'])
        self.assertFalse(t.update(30,ids,scores(flanks+[(3,4)])).exchange_candidates)

    def test_missing_flank_cannot_be_a_four_cell_swap(self):
        flanks = [(1,3),(1,4),(2,3)]
        t = tracker()
        t.update(0,(1,2,3,4),scores(flanks+[(1,2)]))
        t.update(10,(1,2,3,4),scores(flanks+[(3,4)]))
        self.assertFalse(t.update(20,(1,2,3,4),scores(flanks+[(3,4)])).exchange_candidates)

    def test_five_cell_ET_and_TE_compatible_motifs(self):
        ids = (1,2,3,4,5)
        all_edges = list(combinations(ids,2))
        without_face = [pair for pair in all_edges if pair != (4,5)]
        t = tracker()
        t.update(0,ids,scores(without_face))
        t.update(10,ids,scores(all_edges))
        result = t.update(20,ids,scores(all_edges))
        item, = result.three_dimensional_graph_candidates
        self.assertEqual(item['kind'],'ET_compatible_graph_candidate')
        self.assertEqual(item['supporting_triads'],[(1,2,3)])
        self.assertFalse(item['confirmed_topological_transition'])
        t.update(30,ids,scores(without_face))
        result = t.update(40,ids,scores(without_face))
        self.assertEqual(result.three_dimensional_graph_candidates[0]['kind'],'TE_compatible_graph_candidate')

    def test_multiple_pairings_stay_ambiguous(self):
        ids = tuple(range(1,7))
        before = [pair for pair in combinations(ids,2) if pair != (3,4)]
        after = [pair for pair in combinations(ids,2) if pair not in ((1,2),(5,6))]
        t = tracker()
        t.update(0,ids,scores(before))
        t.update(10,ids,scores(after))
        result = t.update(20,ids,scores(after))
        self.assertEqual(len(result.exchange_candidates),2)
        self.assertTrue(all(item['ambiguous_pairing'] for item in result.exchange_candidates))

    def test_simultaneous_motifs_sharing_a_cell_are_not_independent(self):
        ids = tuple(range(1,8))
        flanks = [(1,3),(1,4),(2,3),(2,4),(1,6),(1,7),(5,6),(5,7)]
        t = tracker()
        t.update(0,ids,scores(flanks+[(1,2),(1,5)]))
        t.update(10,ids,scores(flanks+[(3,4),(6,7)]))
        candidates = t.update(20,ids,scores(flanks+[(3,4),(6,7)])).exchange_candidates
        self.assertEqual(len(candidates),2)
        self.assertTrue(all(item['ambiguous_pairing'] for item in candidates))
        self.assertTrue(all(item['overlapping_cells'] == [1] for item in candidates))

    def test_later_alternative_marks_prior_provisional_pairing_ambiguous(self):
        ids = tuple(range(1,7))
        flanks = [(a,b) for a in (1,2) for b in (3,4,5,6)]
        t = tracker()
        t.update(0,ids,scores(flanks+[(1,2)]))
        t.update(10,ids,scores(flanks+[(3,4)]))
        first, = t.update(20,ids,scores(flanks+[(3,4)])).exchange_candidates
        self.assertFalse(first['ambiguous_pairing'])
        t.update(30,ids,scores(flanks+[(3,4),(5,6)]))
        later, = t.update(40,ids,scores(flanks+[(3,4),(5,6)])).exchange_candidates
        self.assertTrue(later['ambiguous_pairing'])
        self.assertEqual(later['prior_pairing_conflicts'],[(first['lost_change'],first['gained_change'])])

    def test_identity_changes_and_duplicate_steps_are_rejected(self):
        t = tracker()
        t.update(0,(1,2),{})
        with self.assertRaises(ValueError):
            t.update(10,(1,3),{})
        with self.assertRaises(ValueError):
            t.update(0,(1,2),{})
        with self.assertRaises(ValueError):
            t.update(10,(1,2),{(1,2):(2.,3.),(2,1):(2.,3.)})


if __name__ == '__main__':
    unittest.main()
