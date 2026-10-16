"""Persistent contact changes and explicitly labelled graph-exchange candidates.

A pairwise graph cannot establish collapse of a tricellular junction. ET/TE
compatibility and four-cell diagonal swaps are therefore candidate annotations,
not confirmed three-dimensional topological transitions.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from itertools import combinations
import math


def edge(a, b):
    if a == b:
        raise ValueError('self contacts are not allowed')
    return tuple(sorted((a, b)))


@dataclass(frozen=True)
class Change:
    index: int
    kind: str
    pair: tuple[int, int]
    last_previous_step: int
    first_changed_step: int
    confirmed_step: int


@dataclass(frozen=True)
class Pending:
    desired: bool
    first_step: int
    previous_step: int
    from_state: bool | None

    def record(self, pair, reason):
        return dict(pair=pair, reason=reason, **asdict(self),
            kind='pending_state_establishment' if self.from_state is None else 'pending_contact_change')


@dataclass(frozen=True)
class Update:
    step: int
    edges: tuple[tuple[int, int], ...]
    changes: tuple[Change, ...]
    exchange_candidates: tuple[dict, ...]
    three_dimensional_graph_candidates: tuple[dict, ...]
    uncertain_pairs: tuple[tuple[int, int], ...]
    pending_count: int
    observation_reset: bool
    unobservable_ids: tuple[int, ...]
    censored_pending: tuple[dict, ...]


class ContactTracker:
    def __init__(self, *, area_on, area_off, persistence_steps,
                 max_frame_gap, exchange_window):
        if (not math.isfinite(area_on) or not math.isfinite(area_off)
                or not 0 <= area_off < area_on):
            raise ValueError('require finite 0 <= area_off < area_on')
        for name, value, minimum in (('persistence_steps', persistence_steps, 0),
                                     ('max_frame_gap', max_frame_gap, 1),
                                     ('exchange_window', exchange_window, 0)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        self.area_on, self.area_off = area_on, area_off
        self.persistence = persistence_steps
        self.max_gap, self.window = max_frame_gap, exchange_window
        self.ids = None
        self.last_step = None
        self.states = {}
        self.present_since = {}
        self.pending = {}
        self.recent = deque()
        self.emitted = set()
        self.last_change = {}
        self.next_change = 0
        self.unobservable = set()

    def _scores(self, ids, scores):
        result = {}
        for pair, bounds in scores.items():
            key = edge(*pair)
            if key in result or any(gid not in ids for gid in key):
                raise ValueError('duplicated pair or unknown persistent ID')
            low, high = bounds
            if not all(math.isfinite(x) for x in bounds) or not 0 <= low <= high:
                raise ValueError('invalid contact-area bounds')
            result[key] = (low, high)
        return result

    def _baseline(self, step, scores):
        self.states.clear()
        self.present_since.clear()
        self.pending.clear()
        self.recent.clear()
        self.emitted.clear()
        self.last_change.clear()
        for pair, (low, high) in scores.items():
            state = True if low >= self.area_on else False if high <= self.area_off else None
            self.states[pair] = state
            if state:
                self.present_since[pair] = step

    def update(self, step, ids, scores, *, unobservable_ids=()):
        ids = tuple(sorted(ids))
        if (len(set(ids)) != len(ids) or not ids
                or any(not isinstance(gid,int) or isinstance(gid,bool) or gid < 0 for gid in ids)):
            raise ValueError('persistent IDs must be unique and nonempty')
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError('accepted step must be a nonnegative integer')
        if self.ids is not None and ids != self.ids:
            raise ValueError('cell identities changed; start a new observation cohort')
        if self.last_step is not None and step <= self.last_step:
            raise ValueError('steps must increase; verify/deduplicate restart endpoints first')
        scores = self._scores(set(ids), scores)
        unobservable = set(unobservable_ids)
        if not unobservable <= set(ids):
            raise ValueError('unobservable cell has an unknown persistent ID')
        if any(unobservable.intersection(pair) for pair in scores):
            raise ValueError('an unobservable cell cannot have measured contact scores')
        reset = self.last_step is None or step-self.last_step > self.max_gap
        changes, uncertain, censored = [], [], []
        if reset:
            censored = [pending.record(pair,'observation_gap') for pair,pending in sorted(self.pending.items())]
            self._baseline(step, scores)
            uncertain = [pair for pair, state in self.states.items() if state is None]
        else:
            pairs = set(self.states) | set(scores) | set(self.pending)
            for pair in sorted(pairs):
                if unobservable.intersection(pair):
                    self.states[pair] = None
                    pending = self.pending.pop(pair, None)
                    if pending is not None:
                        censored.append(pending.record(pair,'unobservable_cell'))
                    self.present_since.pop(pair, None)
                    self.last_change.pop(pair, None)
                    continue
                # A previously unseen pair is known absent at the last frame
                # only if both cells were observable there.
                default = None if self.unobservable.intersection(pair) else False
                state = self.states.get(pair, default)
                self.states.setdefault(pair, state)
                low, high = scores.get(pair, (0.0, 0.0))
                desired = True if low >= self.area_on else False if high <= self.area_off else None
                if desired is None:
                    self.pending.pop(pair, None)
                    straddles = (low <= self.area_off < high if state is True
                                 else low < self.area_on <= high)
                    if state is None or straddles:
                        uncertain.append(pair)
                    continue
                if desired == state:
                    self.pending.pop(pair, None)
                    continue
                pending = self.pending.get(pair)
                if pending is None or pending.desired != desired:
                    pending = Pending(desired, step, self.last_step, state)
                    self.pending[pair] = pending
                if step-pending.first_step < self.persistence:
                    continue
                self.pending.pop(pair)
                self.states[pair] = desired
                if desired:
                    self.present_since[pair] = pending.first_step
                else:
                    self.present_since.pop(pair, None)
                # Unknown initial state is established, not a measured change.
                if state is not None:
                    changes.append(Change(self.next_change,
                        'contact_gained' if desired else 'contact_lost', pair,
                        pending.previous_step, pending.first_step, step))
                    self.last_change[pair] = self.next_change
                    self.next_change += 1
        self.ids, self.last_step, self.unobservable = ids, step, unobservable
        self.recent.extend(changes)
        keep_after = step-self.window-self.persistence-2*self.max_gap
        while self.recent and self.recent[0].confirmed_step < keep_after:
            self.recent.popleft()
        live = {change.index for change in self.recent}
        self.emitted = {pair for pair in self.emitted if set(pair) <= live}
        exchanges = self._four_cell_candidates(changes)
        bulk = self._bulk_candidates(changes)
        return Update(step, tuple(sorted(self.present_since)), tuple(changes),
            tuple(exchanges), tuple(bulk), tuple(sorted(uncertain)),
            len(self.pending), reset, tuple(sorted(unobservable)), tuple(censored))

    def _stable(self, pairs, before_step):
        return all(pair in self.present_since and self.present_since[pair] <= before_step
                   for pair in pairs)

    def _four_cell_candidates(self, new_changes):
        fresh = {change.index for change in new_changes}
        gained = [change for change in self.recent if change.kind == 'contact_gained']
        lost = [change for change in self.recent if change.kind == 'contact_lost']
        candidates = []
        for before in lost:
            for after in gained:
                key = (before.index, after.index)
                if key in self.emitted or not fresh.intersection(key):
                    continue
                if (before.pair in self.present_since or after.pair not in self.present_since
                        or self.last_change.get(before.pair) != before.index
                        or self.last_change.get(after.pair) != after.index):
                    continue
                if set(before.pair).intersection(after.pair):
                    continue
                interval_gap = max(0, max(before.last_previous_step, after.last_previous_step)
                                   - min(before.first_changed_step, after.first_changed_step))
                if interval_gap > self.window:
                    continue
                flanks = [edge(a,b) for a in before.pair for b in after.pair]
                earliest = min(before.last_previous_step, after.last_previous_step)
                if not self._stable(flanks, earliest):
                    continue
                candidates.append(dict(kind='four_cell_graph_exchange_candidate',
                    lost_change=before.index, gained_change=after.index,
                    lost_pair=before.pair, gained_pair=after.pair,
                    cells=sorted(before.pair+after.pair),
                    interval=[min(before.last_previous_step, after.last_previous_step),
                              max(before.first_changed_step, after.first_changed_step)],
                    confirmed_step=self.last_step, stable_flanks=sorted(flanks),
                    confirmed_topological_T1=False))
                self.emitted.add(key)
        use = Counter(index for key in self.emitted for index in key)
        cell_use = Counter(gid for item in candidates for gid in item['cells'])
        current = {(item['lost_change'],item['gained_change']) for item in candidates}
        for item in candidates:
            key = (item['lost_change'],item['gained_change'])
            item['prior_pairing_conflicts'] = sorted(previous for previous in self.emitted-current
                if set(previous).intersection(key))
            item['overlapping_cells'] = [gid for gid in item['cells'] if cell_use[gid] > 1]
            item['ambiguous_pairing'] = (use[item['lost_change']] > 1
                or use[item['gained_change']] > 1 or bool(item['overlapping_cells']))
            item['pairing_is_provisional'] = True
        return candidates

    def _bulk_candidates(self, changes):
        result = []
        neighbours = {gid: set() for gid in self.ids}
        for a,b in self.present_since:
            neighbours[a].add(b)
            neighbours[b].add(a)
        for change in changes:
            a,b = change.pair
            common = sorted(neighbours[a] & neighbours[b])
            triads = []
            truncated = False
            for trio in combinations(common,3):
                required = [edge(x,y) for x,y in combinations(trio,2)]
                required += [edge(x,y) for x in (a,b) for y in trio]
                if self._stable(required, change.last_previous_step):
                    if len(triads) == 128:
                        truncated = True
                        break
                    triads.append(trio)
            if triads:
                result.append(dict(kind='ET_compatible_graph_candidate' if change.kind == 'contact_gained'
                                   else 'TE_compatible_graph_candidate',
                    change=change.index, pair=change.pair, supporting_triads=triads,
                    ambiguous=len(triads)>1 or truncated, alternatives_truncated=truncated,
                    tricellular_junction_measured=False, confirmed_topological_transition=False))
        return result
