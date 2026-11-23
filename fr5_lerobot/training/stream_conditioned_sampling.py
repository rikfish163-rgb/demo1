"""Optional source-balanced sampling of recorded open-gripper approaches.

The caller supplies the existing TransitionSampler class and its already
validated training rows, episodes and provenance. No data or labels are added
or changed here. Keeping the base class as an argument avoids an import cycle
when the training entry point loads this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math


APPROACH_OPEN_VERSION = 'source-transition-motion-approach-open-balanced-batch-v1'
APPROACH_OPEN_FOURTH_SCHEDULE = ('approach_open', 'all', 'hold', 'approach_open', 'all')
EARLY_SWITCH_CONTEXT_VERSION = 'source-transition-motion-approach-open-early-switch-context-balanced-batch-v1'
CUP_TIMING_SAMPLING_VERSION = 'source-balanced-cup-timing-contact-return-v1'
CUP_CLOCK_SAMPLING_VERSION = 'source-balanced-cup-clock-hold-return-v1'
CUP_EVENT_SAMPLING_VERSION = 'source-balanced-camera-clock-cup-events-v1'
CUP_BOUNDARY_SAMPLING_VERSION = 'source-balanced-camera-clock-cup-boundaries-v1'
CUP_SUPPORT_SAMPLING_VERSION = 'source-balanced-camera-clock-cup-support-v1'
CUP_COUPLED_SAMPLING_VERSION = 'source-balanced-camera-clock-cup-coupled-support-v4'
_COUPLED_CLOSE_CONTEXT_OFFSETS = (None, -1, 0, 0)
EARLY_SWITCH_CONTEXT_FOURTH_SCHEDULE = (
    'approach_open', 'before_close', 'hold', 'approach_open', 'all',
    'approach_open', 'before_open', 'hold', 'approach_open', 'all',
)
EARLY_SWITCH_OFFSETS = (-30, -24, -18, -12)
EARLY_SWITCH_CONTEXT_LENGTH = 10
OPEN_CHUNK_LENGTH = 50
MIN_ACTUAL_OPEN = .98


def find_approach_open_anchors(rows, episodes, close_events):
    """Return eligible indices by source and the first real close per episode.

    ``close_events`` must be the base sampler's provenance-checked transition
    events. A complete chunk begins at the anchor and never crosses an episode
    boundary. Episodes without an observed real close contribute no anchors.
    """
    first_close = {}
    for event in close_events:
        ep, position = event['episode'], event['position']
        if ep not in episodes or not 0 <= position < len(episodes[ep]):
            raise ValueError('Close event is outside its episode')
        if episodes[ep][position] != event['index']:
            raise ValueError('Close event index does not match its episode')
        if ep not in first_close or position < first_close[ep]['position']:
            first_close[ep] = event

    eligible = {}
    for ep, indices in episodes.items():
        for index in indices:
            eligible.setdefault(str(rows[index]['_physical_source']), [])
        event = first_close.get(ep)
        if event is None:
            continue
        for position, index in enumerate(indices):
            if position >= event['position']:
                break
            state = rows[index]['observation.state']
            if float(state[7]) != 1. or not MIN_ACTUAL_OPEN <= float(state[6]) <= 1.:
                continue
            chunk = indices[position:position + OPEN_CHUNK_LENGTH]
            if (len(chunk) == OPEN_CHUNK_LENGTH
                    and all(float(rows[i]['action'][6]) == 1. for i in chunk)):
                eligible[str(rows[index]['_physical_source'])].append(index)
    return eligible, {ep: {'position': event['position'], 'index': event['index'],
                           'source': str(event['source'])}
                      for ep, event in first_close.items()}


def _json_value(value):
    """Allow NumPy values in fixtures while keeping the hash deterministic."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Unsupported sampling identity value: {type(value).__name__}')


def _sampling_identity(rows, episodes, proof):
    # Bind state to the ordered inputs which determine every base and new draw,
    # including command evidence and holds even when event indices happen to match.
    source_names = {str(rows[i]['_physical_source']) for indices in episodes.values() for i in indices}
    sources = [source for source in proof['sources'] if str(source['source_index']) in source_names]
    data = {'version': APPROACH_OPEN_VERSION,
            'episodes': [{'episode': ep, 'indices': list(indices),
                          'rows': [{key: rows[i].get(key) for key in
                                    ('index', 'episode_index', 'frame_index', '_physical_source', '_split',
                                     '_grid_time', '_gripper_target_ref', 'observation.state', 'action')}
                                   for i in indices]}
                         for ep, indices in episodes.items()],
            'sources': sources}
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False,
                         default=_json_value).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def approach_open_sampler_class(base_sampler_class):
    """Create an optional sampler subclass without importing the trainer.

    Instantiate the returned class with the same arguments as the supplied
    TransitionSampler. Its balanced draws retain the base close/open/motion
    slots and use APPROACH_OPEN_FOURTH_SCHEDULE for the fourth slot.
    """
    class ApproachOpenSampler(base_sampler_class):
        VERSION = APPROACH_OPEN_VERSION
        FOURTH_SCHEDULE = APPROACH_OPEN_FOURTH_SCHEDULE

        def __init__(self, rows, episodes, proof, seed):
            sources = {str(s['source_index']): s for s in proof['sources']}
            for indices in episodes.values():
                for i in indices:
                    source = str(rows[i]['_physical_source'])
                    if (rows[i].get('_split') == 'validation'
                            or sources[source].get('split') == 'validation'):
                        raise ValueError('Approach-open sampling requires training episodes only')
            super().__init__(rows, episodes, proof, seed)
            self.approach_open_by_source, self.first_close_by_episode = find_approach_open_anchors(
                rows, episodes, self.events['close'])
            if not any(self.approach_open_by_source.values()):
                raise ValueError('No eligible approach-open anchors in training episodes')
            self._approach_sources = {k: v for k, v in self.approach_open_by_source.items() if v}
            self._approach_draws = {k: 0 for k in self.approach_open_by_source}
            self._approach_seen = {k: set() for k in self.approach_open_by_source}
            self.category_counts['approach_open'] = 0
            self.sampling_identity = _sampling_identity(rows, episodes, proof)

        def draw_category(self, category):
            if category != 'approach_open':
                return super().draw_category(category)
            index = self._pick_source_frame(self._approach_sources)
            source = str(self.rows[index]['_physical_source'])
            self.category_counts[category] += 1
            self._approach_draws[source] += 1
            self._approach_seen[source].add(index)
            return index, {'category': category, 'source': source}

        def getstate(self):
            state = super().getstate()
            state.update({'sampling_identity': self.sampling_identity,
                          'approach_open_draws_by_source': dict(self._approach_draws),
                          'approach_open_seen_by_source': {k: sorted(v) for k, v in self._approach_seen.items()}})
            return state

        def setstate(self, state):
            if not isinstance(state, dict) or state.get('version') != self.VERSION:
                raise ValueError('Sampler state belongs to a different sampling strategy')
            if state.get('sampling_identity') != self.sampling_identity:
                raise ValueError('Sampler identity differs from the resumed dataset')
            draws = state.get('approach_open_draws_by_source')
            seen = state.get('approach_open_seen_by_source')
            counts = state.get('category_counts')
            keys = set(self.approach_open_by_source)
            if (not isinstance(draws, dict) or not isinstance(seen, dict)
                    or set(draws) != keys or set(seen) != keys or not isinstance(counts, dict)
                    or set(counts) != set(self.category_counts)):
                raise ValueError('Invalid approach-open sampler state')
            restored_seen = {}
            for source in keys:
                values = seen[source]
                if (type(draws[source]) is not int or draws[source] < 0
                        or not isinstance(values, (list, tuple))
                        or any(type(index) is not int for index in values)):
                    raise ValueError('Invalid approach-open sampler state')
                restored_seen[source] = set(values)
                if (len(restored_seen[source]) != len(values)
                        or not restored_seen[source].issubset(self.approach_open_by_source[source])
                        or len(values) > draws[source]
                        or (draws[source] > 0 and not values)):
                    raise ValueError('Invalid approach-open sampler state')
            if counts.get('approach_open') != sum(draws.values()):
                raise ValueError('Approach-open draw counts do not match sampler state')
            super().setstate(state)
            self._approach_draws = dict(draws)
            self._approach_seen = restored_seen

        def report(self):
            result = copy.deepcopy(super().report())
            result['balanced_group'] = ['close', 'open', 'motion', 'approach_open_all_or_recorded_hold']
            result['sampling_identity'] = self.sampling_identity
            result['approach_open'] = {
                'chunk_length': OPEN_CHUNK_LENGTH,
                'minimum_actual_gripper': MIN_ACTUAL_OPEN,
                'gripper_feedback_valid_required': True,
                'all_chunk_gripper_targets': 1.,
                'selection': 'uniform_over_sources_with_eligible_anchors_then_uniform_within_source',
                'eligible': sum(len(v) for v in self.approach_open_by_source.values()),
                'draws': sum(self._approach_draws.values()),
                'sampled_unique': sum(len(v) for v in self._approach_seen.values()),
                'per_source': {source: {'eligible': len(indices), 'draws': self._approach_draws[source],
                                        'sampled_unique': len(self._approach_seen[source])}
                               for source, indices in sorted(self.approach_open_by_source.items())},
                'first_close_by_episode': copy.deepcopy(self.first_close_by_episode),
            }
            return result

    return ApproachOpenSampler


def find_early_switch_context_anchors(rows, episodes, events):
    """Group eligible original-label anchors by their recorded switch event."""
    eligible = {'before_close': [], 'before_open': []}
    for kind in ('close', 'open'):
        for event in events[kind]:
            indices = episodes[event['episode']]
            old = float(event['old'])
            anchors = []
            for offset in EARLY_SWITCH_OFFSETS:
                pos = event['position'] + offset
                start, end = pos - EARLY_SWITCH_CONTEXT_LENGTH, pos + EARLY_SWITCH_CONTEXT_LENGTH
                if (start >= 0 and end <= len(indices)
                        and all(float(rows[i]['action'][6]) == old for i in indices[start:end])):
                    anchors.append((offset, indices[pos]))
            if anchors:
                eligible[f'before_{kind}'].append({'event': event, 'anchors': anchors})
    return eligible


def early_switch_context_sampler_class(base_sampler_class):
    """Extend approach-open sampling with balanced pre-switch examples."""
    approach_sampler = approach_open_sampler_class(base_sampler_class)

    class EarlySwitchContextSampler(approach_sampler):
        VERSION = EARLY_SWITCH_CONTEXT_VERSION
        FOURTH_SCHEDULE = EARLY_SWITCH_CONTEXT_FOURTH_SCHEDULE

        def __init__(self, rows, episodes, proof, seed):
            super().__init__(rows, episodes, proof, seed)
            self.early_switch_events = find_early_switch_context_anchors(rows, episodes, self.events)
            if not all(self.early_switch_events.values()):
                raise ValueError('Early-switch sampling requires eligible close and open events')
            self.pending_early_events = {kind: [] for kind in self.early_switch_events}
            self.category_counts.update({kind: 0 for kind in self.early_switch_events})

        def draw_category(self, category):
            if category not in self.early_switch_events:
                return super().draw_category(category)
            queue = self.pending_early_events[category]
            events = self.early_switch_events[category]
            if not queue:
                queue.extend(range(len(events)))
                self.rng.shuffle(queue)
            selected = events[queue.pop()]
            offset, index = self.rng.choice(selected['anchors'])
            event = selected['event']
            self.category_counts[category] += 1
            return index, {'category': category, 'event_index': event['index'],
                           'source': event['source'], 'offset_frames': offset}

        def getstate(self):
            state = super().getstate()
            state['pending_early_events'] = copy.deepcopy(self.pending_early_events)
            return state

        def setstate(self, state):
            super().setstate(state)
            self.pending_early_events = copy.deepcopy(state['pending_early_events'])

        def report(self):
            result = super().report()
            result['balanced_group'][-1] = 'approach_open_early_switch_all_or_recorded_hold'
            result['early_switch_context'] = {
                'offset_frames': list(EARLY_SWITCH_OFFSETS),
                'original_gripper_steps_before_and_from_anchor': EARLY_SWITCH_CONTEXT_LENGTH,
                'selection': 'shuffled_events_then_uniform_eligible_offset',
                'eligible_events': {kind: len(events) for kind, events in self.early_switch_events.items()},
                'eligible_anchors': {kind: sum(len(entry['anchors']) for entry in events)
                                     for kind, events in self.early_switch_events.items()},
            }
            return result

    return EarlySwitchContextSampler


def cup_timing_sampler_class(base_sampler_class, *, focus_hold_return=False, camera_events=False,
                             camera_boundaries=False, support_focus=False, coupled_support=False):
    """Train open approaches, steady holds and table-height release together."""
    class CupTimingSampler(base_sampler_class):
        VERSION = (CUP_COUPLED_SAMPLING_VERSION if coupled_support else CUP_SUPPORT_SAMPLING_VERSION if support_focus else
                   CUP_BOUNDARY_SAMPLING_VERSION if camera_boundaries else CUP_EVENT_SAMPLING_VERSION if camera_events else
                   CUP_CLOCK_SAMPLING_VERSION if focus_hold_return else CUP_TIMING_SAMPLING_VERSION)
        FOURTH_SCHEDULE = ('near_open', 'hover_hold', 'return_hold', 'retreat_open', 'all')
        HOVER_ROTATION = (('hover_steady', 'hover_exit', 'hover_deadline', 'hover_deadline') if support_focus else
                          ('hover_steady', 'hover_exit', 'hover_steady'))
        CONTEXT_ROTATION = ('motion', 'return_hold', 'motion', 'retreat_open')
        CAMERA_CONTEXT_ROTATION = (
            ('near_open', 'motion', 'pickup_open_steady', 'all',
             'return_hold', 'retreat_open', 'pickup_open_steady', 'all') if coupled_support else
            ('near_open', 'motion', 'return_hold', 'retreat_open'))

        def __init__(self, rows, episodes, proof, seed):
            super().__init__(rows, episodes, proof, seed)
            if any(rows[i].get('_cup_timing') is None for indices in episodes.values() for i in indices):
                raise ValueError('Cup timing sampling requires causal tracked observations')
            setdown = 'setdown_age_s' in rows[next(iter(episodes.values()))[0]]['_cup_timing']
            self.setdown = setdown
            self.timing_groups = {k: {} for k in self.FOURTH_SCHEDULE if k != 'all'}
            if focus_hold_return:
                self.timing_groups.update({k: {} for k in ('hover_steady', 'hover_exit', 'return_before_open')})
            if support_focus:
                self.timing_groups.update({k: {} for k in ('hover_deadline', 'return_near_support')})
            if coupled_support:
                self.timing_groups['pickup_open_steady'] = {}
            for ep, indices in episodes.items():
                source = str(rows[indices[0]]['_physical_source'])
                opens = [e['position'] for e in self.events['open'] if e['episode'] == ep]
                closes = [e['position'] for e in self.events['close'] if e['episode'] == ep]
                for pos, i in enumerate(indices):
                    row = rows[i]
                    phase = row['_cup_timing']['phase']
                    aligned = row['_camera_aligned_actions'][:10] if camera_boundaries else None
                    targets = (aligned[:, 6].tolist() if aligned is not None else
                               [float(rows[j]['action'][6]) for j in indices[pos:pos + 10]])
                    group = None
                    if len(targets) == 10 and all(x >= .98 for x in targets):
                        if any(10 <= close - pos <= 42 for close in closes):
                            group = 'near_open'
                        elif any(3 <= pos - opened <= 60 for opened in opens):
                            group = 'retreat_open'
                    if len(targets) == 10 and all(x <= .88 for x in targets):
                        if phase == 'hover':
                            group = 'hover_hold'
                        elif phase in (('returning', 'at_table_height') if camera_boundaries else
                                      ('grasp_contact', 'returning')):
                            group = 'return_hold'
                    if group is not None:
                        self.timing_groups[group].setdefault(source, []).append(i)
                    if coupled_support:
                        state = row['_cup_relative_state']
                        prefix = row['_camera_aligned_actions'][:3]
                        if (phase == 'at_pickup' and state[7] == 1. and state[6] >= .98
                                and len(prefix) == 3 and all(target[6] >= .98 for target in prefix)
                                and any(0 < close - pos <= 12 for close in closes)
                                and max(math.dist(target[:3], state[:3]) * 1000. for target in prefix) <= .2):
                            self.timing_groups['pickup_open_steady'].setdefault(source, []).append(i)
                    if not focus_hold_return or len(targets) != 10 or not row['_cup_relative_state'][7]:
                        continue
                    phase_group = None
                    if phase == 'hover' and all(x <= .88 for x in targets):
                        z = row['_cup_relative_state'][2]
                        dz = (((aligned[:, :3] - row['_cup_relative_state'][:3]) @ row['_cup_table_normal']) * 1000.
                              if aligned is not None else
                              [(rows[j]['_cup_relative_action'][2] - z) * 1000. for j in indices[pos:pos + 10]])
                        if max(map(abs, dz)) <= 1.5:
                            phase_group = 'hover_steady'
                        elif min(dz) <= -2.:
                            phase_group = 'hover_exit'
                    elif phase in ('returning', 'at_table_height') and targets[0] <= .88:
                        first_open = next((tick for tick, x in enumerate(targets, 1) if x >= .98), None)
                        if (first_open is not None and 5 <= first_open <= 10 if camera_boundaries else
                                any(2 <= opened - pos <= 12 for opened in opens)):
                            phase_group = 'return_before_open'
                    if phase_group is not None:
                        self.timing_groups[phase_group].setdefault(source, []).append(i)
                    if support_focus and row['_cup_tracking']['native_contact'] == 1:
                        remaining = row['_recorded_hold_remaining_s']
                        gap = row['_cup_timing']['support_gap_mm']
                        if (phase == 'hover' and remaining is not None and .30 <= remaining <= 1.75
                                and all(x <= .88 for x in targets) and abs(dz[0]) <= 1.25):
                            self.timing_groups['hover_deadline'].setdefault(source, []).append(i)
                        if (phase == 'returning' and gap is not None and 2.5 <= gap <= 18. and targets[0] <= .88):
                            self.timing_groups['return_near_support'].setdefault(source, []).append(i)
                        if setdown and phase in ('returning', 'at_table_height'):
                            age = row['_cup_timing']['setdown_age_s']
                            if age is not None and age >= .30 and any(x >= .98 for x in targets[:2]):
                                self.timing_groups.setdefault('settled_open', {}).setdefault(source, []).append(i)
            if not all(self.timing_groups.values()):
                raise ValueError('Expected actual open approaches, held cups and open retreats')
            if setdown and 'settled_open' not in self.timing_groups:
                raise ValueError('No original opening target paired with a stable setdown')
            self.category_counts.update({k: 0 for k in self.timing_groups})
            self.camera_switches = {'close': {}, 'open': {}}
            if camera_events:
                for name in self.camera_switches:
                    for event in self.events[name]:
                        indices = self.episodes[event['episode']]
                        start = event['position']
                        candidates = []
                        for pos in range(max(0, start-4), min(start+9, len(indices))):
                            row = rows[indices[pos]]
                            if not row['_cup_relative_state'][7]:
                                continue
                            state = row['_cup_relative_state'][6]
                            gripper = row['_camera_aligned_actions'][:3]
                            if (name == 'close' and state >= .98) or (name == 'open' and state < .88):
                                tick = next((tick for tick, target in enumerate(gripper, 1)
                                             if (target[6] < .88 if name == 'close' else target[6] >= .98)), None)
                                if tick is not None:
                                    candidates.append((tick, pos))
                        if not candidates:
                            raise ValueError(f'No valid camera-clock {name} observation for source {event["source"]}')
                        self.camera_switches[name][event['index']] = min(candidates)[1]
            self.far_close_context = {}
            if coupled_support:
                for event in self.events['close']:
                    indices = self.episodes[event['episode']]
                    switch = self.camera_switches['close'][event['index']]
                    reference = rows[indices[switch]]['_cup_relative_state'][:3]
                    previous_open = max((e['position'] for e in self.events['open']
                                         if e['episode'] == event['episode'] and e['position'] < switch),
                                        default=-1)
                    candidates = []
                    for position in range(previous_open + 1, switch):
                        row = rows[indices[position]]
                        state = row['_cup_relative_state']
                        targets = row['_camera_aligned_actions'][:3]
                        if not state[7] or state[6] < .98 or len(targets) != 3 or any(a[6] < .98 for a in targets):
                            continue
                        delta = [(x - y) * 1000. for x, y in zip(state[:3], reference)]
                        normal = row['_cup_table_normal']
                        height = float(sum(x * n for x, n in zip(delta, normal)))
                        horizontal = math.sqrt(max(0., sum(x * x for x in delta) - height * height))
                        if horizontal > 15. or abs(height) > 12.:
                            candidates.append((position, horizontal, height))
                    if not candidates:
                        raise ValueError(f'No recorded far open approach for source {event["source"]}')
                    position, horizontal, height = candidates[-1]
                    self.far_close_context[event['index']] = {
                        'position': position, 'index': indices[position], 'source': event['source'],
                        'offset_frames': position - switch,
                        'horizontal_from_close_mm': horizontal, 'normal_from_close_mm': height,
                    }
            self.coverage_sources = {'all': [], 'pickup_open_steady': []} if coupled_support else {}
            self.coverage_frames = {name: {} for name in self.coverage_sources}

        def _draw_coverage(self, category):
            table = self.by_source if category == 'all' else self.timing_groups[category]
            sources = self.coverage_sources[category]
            if not sources:
                sources.extend(sorted(table))
                self.rng.shuffle(sources)
            source = sources.pop()
            frames = self.coverage_frames[category].setdefault(source, [])
            if not frames:
                frames.extend(table[source])
                self.rng.shuffle(frames)
            index = frames.pop()
            self.category_counts[category] += 1
            return index, {'category': category, 'source': source}

        def draw_category(self, category):
            if category in self.coverage_sources:
                return self._draw_coverage(category)
            if category not in self.timing_groups:
                return super().draw_category(category)
            index = self._pick_source_frame(self.timing_groups[category])
            self.category_counts[category] += 1
            return index, {'category': category, 'source': self.rows[index]['_physical_source']}

        def draw_balanced(self, count):
            if count < 4 or count % 4:
                raise ValueError('Balanced effective batch must be a positive multiple of four')
            result = []
            for _ in range(count // 4):
                i = self.fourth_index
                self.fourth_index += 1
                fourth = self.FOURTH_SCHEDULE[i % len(self.FOURTH_SCHEDULE)]
                if camera_events:
                    group = [self.draw_category(self.HOVER_ROTATION[i % len(self.HOVER_ROTATION)]),
                             self.draw_category(self.CAMERA_CONTEXT_ROTATION[i % len(self.CAMERA_CONTEXT_ROTATION)])]
                elif focus_hold_return:
                    group = [self.draw_category(self.HOVER_ROTATION[i % len(self.HOVER_ROTATION)]),
                             self.draw_category(self.CONTEXT_ROTATION[i % len(self.CONTEXT_ROTATION)])]
                else:
                    group = [self.draw_category('motion'), self.draw_category(fourth)]
                for name in ('close', 'open'):
                    event_tick = i
                    if focus_hold_return:
                        if name == 'close' and i % 4 == 3:
                            group.append(self.draw_category('near_open'))
                            continue
                        if name == 'open' and i % 2 == 1:
                            preceding = ('return_near_support' if support_focus else 'return_hold') if (
                                camera_boundaries and i % 4 == 3) else 'return_before_open'
                            group.append(self.draw_category(preceding))
                            continue
                        if name == 'open' and self.setdown:
                            group.append(self.draw_category('settled_open'))
                            continue
                        event_tick = (i - i // 4 if name == 'close' else i // 2)
                    event_index = (event_tick // 4) % len(self.events[name])
                    event = self.events[name][event_index]
                    indices = self.episodes[event['episode']]
                    # Include unchanged state immediately before the real
                    # switch, at the real switch, and just after it.
                    offset = (-2, 0, -1, 1)[event_tick % 4] if name == 'close' else (-2, 0, 0, -1)[event_tick % 4]
                    if camera_events:
                        switch = self.camera_switches[name][event['index']]
                        offset = (_COUPLED_CLOSE_CONTEXT_OFFSETS[event_tick % 4]
                                  if coupled_support and name == 'close'
                                  else 0 if event_tick % 4 != 0 else -2)
                        position = (self.far_close_context[event['index']]['position'] if offset is None
                                    else switch + offset)
                    else:
                        position = event['position'] + offset
                    if position < 0 or position >= len(indices):
                        position = event['position']
                    stats = self.event_counts[name][str(event['index'])]
                    stats['total'] += 1
                    stats['at'] += position == (switch if camera_events else event['position'])
                    self.category_counts[name] += 1
                    group.append((indices[position], {'category': name, 'source': event['source'],
                                 'event_index': event['index'],
                                 'offset_frames': position - (switch if camera_events else event['position'])}))
                self.rng.shuffle(group)
                result.extend(group)
            return result

        def getstate(self):
            state = super().getstate()
            if coupled_support:
                state['coverage_sources'] = copy.deepcopy(self.coverage_sources)
                state['coverage_frames'] = copy.deepcopy(self.coverage_frames)
            return state

        def setstate(self, state):
            super().setstate(state)
            if coupled_support:
                self.coverage_sources = copy.deepcopy(state['coverage_sources'])
                self.coverage_frames = copy.deepcopy(state['coverage_frames'])

        def report(self):
            result = super().report()
            result['balanced_group'][-1] = 'near_open_hover_return_retreat_or_all'
            result['timing_groups'] = {k: sum(map(len, v.values())) for k, v in self.timing_groups.items()}
            result['switch_offsets'] = {'close': [-2, 0, -1, 1], 'open': [-2, 0, 0, -1]}
            if focus_hold_return:
                result['balanced_group'] = ['close_or_open_approach', 'open_or_hold_before_open',
                                            'steady_hover_or_hover_exit', 'motion_return_or_retreat']
                result['hover_rotation'] = list(self.HOVER_ROTATION)
                result['context_rotation'] = list(self.CONTEXT_ROTATION)
            if camera_events:
                result.pop('fourth_rotation')
                result['context_rotation'] = list(self.CAMERA_CONTEXT_ROTATION)
                result['balanced_group'][-1] = 'camera_context'
                result['switch_offsets'] = {'close': ['far_recorded_open', -1, 0, 0] if coupled_support else [-2, 0, 0, 0],
                                            'open': [-2, 0, 0, 0]}
                result['camera_switch_anchors'] = self.camera_switches
                if coupled_support:
                    result['far_close_context'] = copy.deepcopy(self.far_close_context)
                    result['coverage_sampling'] = {
                        'categories': list(self.coverage_sources),
                        'selection': 'shuffled_source_rounds_then_shuffled_frames_without_replacement_per_source',
                        'pickup_open_steady': {
                            'phase': 'at_pickup', 'valid_open_feedback': True,
                            'recorded_prefix_points': 3, 'maximum_recorded_displacement_mm': .2,
                            'maximum_frames_before_close': 12,
                            'eligible_by_source': {source: len(indices) for source, indices in
                                                   self.timing_groups['pickup_open_steady'].items()},
                        },
                    }
            if camera_boundaries:
                result['timing_target_clock'] = 'camera_monotonic_plus_tick/6'
                result['return_gripper_rotation'] = ['settled_open' if self.setdown else 'switch', 'return_before_open',
                                                     'settled_open' if self.setdown else 'switch',
                                                     'return_near_support' if support_focus else 'return_hold']
            return result

    return CupTimingSampler
