"""CPU-only tests for the optional approach-open sampling profile."""
import copy
import json
import pickle
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import stream_conditioned_sampling as conditioned
import train_stream_pi05 as training


def fixture():
    # Source 0 has several episodes, including one without a close. Source 3
    # has real transitions but no full approach-open window; 2 is held out.
    specs = [(0, 160, 100, 130), (1, 260, 200, 230), (0, 90, 70, 80),
             (0, 65, None, None), (3, 90, 30, 60), (2, 160, 100, 130)]
    rows = []
    for episode, (source, length, close, opened) in enumerate(specs):
        for position in range(length):
            target = .76 if close is not None and close <= position < opened else 1.
            value = position / 100.
            rows.append({'index': len(rows), 'episode_index': episode, 'frame_index': position,
                         'timestamp': position / 6, '_physical_source': str(source),
                         '_split': 'validation' if source == 2 else 'train',
                         '_grid_time': float(len(rows)),
                         '_gripper_target_ref': {'kind': 'gripper', 'executed': True,
                             'source': 'wire_command', 'sequence': position,
                             'target_degrees_percent': [0.] * 6 + [100 * target]},
                         'observation.state': [value] * 6 + [target, 1.],
                         'action': [value + .01] * 6 + [target]})
    proof = {'sources': [{'source_index': i, 'split': 'validation' if i == 2 else 'train',
                          'source_scope_sha256': f'scope-{i}', 'user_review_sha256': f'review-{i}',
                          'hold_records': [{'start_monotonic': 0., 'end_monotonic': 10000.}]}
                         for i in range(4)]}
    episodes = training.validate_rows(rows)
    train, validation, _ = training.split_sources(rows, episodes)
    return rows, proof, train, validation


ApproachOpenSampler = conditioned.approach_open_sampler_class(training.TransitionSampler)
EarlySwitchContextSampler = conditioned.early_switch_context_sampler_class(training.TransitionSampler)


def camera_fixture():
    import numpy as np

    rows, proof, train, validation = fixture()
    events = training.transition_events(rows, train)
    for ep, indices in train.items():
        close = next((e['position'] for e in events['close'] if e['episode'] == ep), None)
        opened = next((e['position'] for e in events['open'] if e['episode'] == ep), None)
        for pos, index in enumerate(indices):
            row = rows[index]
            held = close is not None and close <= pos < opened
            returning = held and pos >= opened - 14
            phase = 'returning' if returning else 'hover' if held and pos >= close + 4 else 'approaching'
            if close is not None and close - 8 <= pos < close:
                phase = 'at_pickup'
            z = .09 if returning else .1
            feedback = [0., 0.] if pos == close else row['observation.state'][6:]
            x = .004 * max(0, close - pos - 8) if close is not None else 0.
            row['_cup_relative_state'] = [x, 0., z, 0., 0., 0., *feedback]
            row['_cup_timing'] = {'phase': phase, 'support_gap_mm': 5.}
            row['_cup_tracking'] = {'native_contact': int(held)}
            row['_recorded_hold_remaining_s'] = 1.
            row['_cup_table_normal'] = np.array([0., 0., 1.], dtype=np.float32)
            row['_camera_aligned_actions'] = np.array([
                [.004 * max(0, close - tick - 8) if close is not None else 0., 0.,
                 .09 if close is not None and close <= tick < opened and tick >= opened - 14 else .1,
                 0., 0., 0., rows[indices[min(tick, len(indices) - 1)]]['action'][6]]
                for tick in range(pos + 1, pos + 11)
            ])
    return rows, proof, train, validation


class CupTimingSamplingTests(unittest.TestCase):
    def test_coupled_close_samples_valid_open_context_and_real_switch(self):
        rows, proof, train, validation = camera_fixture()
        events = training.transition_events(rows, train)
        original_actions = [tuple(row['action']) for row in rows]
        sampler = training.make_sampler(training.CUP_COUPLED_SAMPLING_PROFILE, rows, train, proof, seed=42)
        json.dumps(sampler.report(), allow_nan=False)
        picks = [p for _ in range(8 * len(events['close'])) for p in sampler.draw_balanced(4)]
        held_out = {i for indices in validation.values() for i in indices}
        closed = [(index, info) for index, info in picks if info['category'] == 'close']
        self.assertTrue(all(index not in held_out for index, _ in picks))
        for event in events['close']:
            selected = [(index, info['offset_frames']) for index, info in closed
                        if info['event_index'] == event['index']]
            offsets = {offset for _, offset in selected}
            self.assertEqual(offsets & {-1, 0}, {-1, 0})
            self.assertEqual(len(offsets), 3)
            far = sampler.far_close_context[event['index']]
            self.assertLess(far['offset_frames'], -3)
            self.assertTrue(far['horizontal_from_close_mm'] > 15. or abs(far['normal_from_close_mm']) > 12.)
            self.assertTrue(all(rows[index]['_cup_relative_state'][6:] == [1., 1.]
                                for index, _ in selected))
            for index, offset in selected:
                target = rows[index]['_camera_aligned_actions'][0, 6]
                self.assertEqual(target, .76 if offset == 0 else 1.)
        self.assertEqual([tuple(row['action']) for row in rows], original_actions)

    def test_camera_context_covers_recorded_static_pickup_and_unrestricted_inputs(self):
        rows, proof, train, validation = camera_fixture()
        event = training.transition_events(rows, train)['close'][0]
        missed = train[event['episode']][event['position'] - 5]
        rows[missed]['action'][:6] = rows[missed]['observation.state'][:6]
        sampler = training.make_sampler(training.CUP_COUPLED_SAMPLING_PROFILE, rows, train, proof, seed=42)
        self.assertNotIn(missed, sampler.motion['0'])
        self.assertNotIn(missed, sampler.timing_groups['near_open'].get('0', []))
        self.assertIn(missed, sampler.timing_groups['pickup_open_steady']['0'])
        original = [row['_camera_aligned_actions'].copy() for ep in train.values() for row in (rows[i] for i in ep)]
        picks = [p for _ in range(64) for p in sampler.draw_balanced(4)]
        self.assertTrue(any(i == missed and info['category'] == 'pickup_open_steady' for i, info in picks))
        self.assertEqual(sampler.category_counts['all'], 16)
        self.assertEqual(sampler.category_counts['pickup_open_steady'], 16)
        heldout = {i for indices in validation.values() for i in indices}
        self.assertTrue(all(i not in heldout for i, _ in picks))
        for recorded, i in zip(original, (i for indices in train.values() for i in indices)):
            self.assertTrue((recorded == rows[i]['_camera_aligned_actions']).all())

    def test_coverage_draws_visit_every_source_frame_and_resume_mid_round(self):
        rows, proof, train, _ = camera_fixture()
        sampler = training.make_sampler(training.CUP_COUPLED_SAMPLING_PROFILE, rows, train, proof, seed=9)
        table = sampler.by_source
        rounds = max(map(len, table.values()))
        selected = [sampler.draw_category('all')[0] for _ in range(rounds * len(table))]
        self.assertEqual(set(selected), {i for indices in train.values() for i in indices})
        for _ in range(9):
            sampler.draw_balanced(4)
        state = pickle.loads(pickle.dumps(sampler.getstate()))
        restored = training.make_sampler(training.CUP_COUPLED_SAMPLING_PROFILE, rows, train, proof, seed=99)
        restored.setstate(state)
        self.assertEqual([sampler.draw_balanced(4) for _ in range(32)],
                         [restored.draw_balanced(4) for _ in range(32)])

    def test_boundary_and_negative_examples_use_original_actions_and_training_sources(self):
        rows, proof, train, validation = fixture()
        events = training.transition_events(rows, train)
        for row in rows:
            row['_cup_timing'] = {'phase': 'approaching'}
        for close in events['close']:
            for index in train[close['episode']][close['position']:close['position']+5]:
                rows[index]['_cup_timing'] = {'phase': 'grasp_contact'}
            for index in train[close['episode']][close['position']+5:close['position']+20]:
                rows[index]['_cup_timing'] = {'phase': 'hover'}
        actions = [tuple(row['action']) for row in rows]
        sampler = training.make_sampler(training.CUP_TIMING_SAMPLING_PROFILE, rows, train, proof, seed=41)
        groups = [sampler.draw_balanced(4) for _ in range(4 * len(events['close']))]
        selected = [item for group in groups for item in group]
        held_out = {i for indices in validation.values() for i in indices}
        self.assertTrue(all(i not in held_out for i, _ in selected))
        for category in ('near_open', 'hover_hold', 'return_hold', 'retreat_open'):
            self.assertTrue(any(info['category'] == category for _, info in selected))
        for kind in ('open', 'close'):
            matched = {i for i, info in selected if info['category'] == kind and info['offset_frames'] == 0}
            self.assertEqual(matched, {e['index'] for e in events[kind]})
        self.assertEqual([tuple(row['action']) for row in rows], actions)

    def test_clock_sampling_keeps_both_sides_of_motion_and_gripper_boundaries(self):
        rows, proof, train, validation = fixture()
        events = training.transition_events(rows, train)
        for row in rows:
            row['_cup_timing'] = {'phase': 'approaching'}
            row['_cup_relative_action'] = [0., 0., .1, 0., 0., 0., row['action'][6]]
            row['_cup_relative_state'] = [0., 0., .1, 0., 0., 0., *row['observation.state'][6:]]
        for event in events['close']:
            indices, pos = train[event['episode']], event['position']
            for i in indices[pos:pos+5]:
                rows[i]['_cup_timing']['phase'] = 'grasp_contact'
            for i in indices[pos+5:pos+20]:
                rows[i]['_cup_timing']['phase'] = 'hover'
            for i in indices[pos+20:pos+28]:
                rows[i]['_cup_timing']['phase'] = 'returning'
                rows[i]['_cup_relative_action'][2] = .09
        actions = [tuple(row['_cup_relative_action']) for row in rows]
        profile = training.resolve_sampling_profile(training.CUP_CLOCK_SAMPLING_PROFILE)
        sampler = training.make_sampler(profile, rows, train, proof, seed=11)
        picks = [p for _ in range(8 * len(events['open'])) for p in sampler.draw_balanced(4)]
        eligible = {i for source in train.values() for i in source}
        self.assertTrue(all(i in eligible for i, _ in picks))
        selected = {info['category'] for _, info in picks}
        self.assertTrue({'hover_steady', 'hover_exit', 'return_before_open', 'close', 'open'} <= selected)
        self.assertEqual({i for i, info in picks if info['category']=='open' and info['offset_frames']==0},
                         {e['index'] for e in events['open']})
        self.assertEqual([tuple(row['_cup_relative_action']) for row in rows], actions)

        for row in rows:
            row['_camera_aligned_actions'] = [row['_cup_relative_action'].copy()]
        for kind, group in events.items():
            for event in group:
                at = train[event['episode']][event['position']]
                before = train[event['episode']][event['position']-1]
                rows[at]['_camera_aligned_actions'][0][6] = event['old']
                if event == group[0]:
                    second = rows[before]['_camera_aligned_actions'][0].copy()
                    second[6] = event['new']
                    rows[before]['_camera_aligned_actions'].append(second)
                else:
                    rows[before]['_camera_aligned_actions'][0][6] = event['new']
        aligned = training.make_sampler(training.CUP_EVENT_SAMPLING_PROFILE, rows, train, proof, seed=12)
        for kind in ('open', 'close'):
            self.assertEqual(aligned.camera_switches[kind], {e['index']:e['position']-1 for e in events[kind]})
        self.assertTrue(all(i in eligible for _ in range(10) for i, _ in aligned.draw_balanced(4)))


class ApproachOpenSamplingTests(unittest.TestCase):
    def test_before_first_real_close_per_episode_and_valid_actual_feedback(self):
        rows, proof, train, validation = fixture()
        first = train[0]
        rows[first[2]]['observation.state'][6:] = [.98, 1.]
        rows[first[3]]['observation.state'][6:] = [.979, 1.]
        rows[first[4]]['observation.state'][6:] = [1., 0.]
        # A later close in episode 0 must not expand eligibility after its first.
        for index in first[150:]:
            rows[index]['action'][6] = .76
            rows[index]['_gripper_target_ref']['target_degrees_percent'][6] = 76.
        labels_before = [list(row['action']) for row in rows]
        sampler = ApproachOpenSampler(rows, train, proof, seed=7)
        eligible = sampler.approach_open_by_source
        self.assertIn(first[2], eligible['0'])
        self.assertNotIn(first[3], eligible['0'])
        self.assertNotIn(first[4], eligible['0'])
        self.assertIn(first[50], eligible['0'])
        self.assertNotIn(first[51], eligible['0'])
        self.assertFalse(set(first[100:]) & set(eligible['0']))
        self.assertEqual(sampler.first_close_by_episode[0]['index'], first[100])
        self.assertEqual(sampler.first_close_by_episode[2]['index'], train[2][70])
        self.assertFalse(set(train[3]) & set(eligible['0']))
        self.assertNotIn('2', eligible)
        self.assertFalse(set(validation[5]) & {i for values in eligible.values() for i in values})
        self.assertEqual([row['action'] for row in rows], labels_before)

    def test_all_fifty_recorded_targets_must_be_exactly_open(self):
        rows, proof, train, _ = fixture()
        first = train[0]
        # Too small to be classified as a target transition by the base sampler,
        # but still not a 100% label. It invalidates offsets 0 through 49.
        rows[first[49]]['action'][6] = .999999
        sampler = ApproachOpenSampler(rows, train, proof, seed=8)
        eligible = set(sampler.approach_open_by_source['0'])
        self.assertFalse(set(first[:50]) & eligible)
        self.assertIn(first[50], eligible)
        self.assertNotIn(first[51], eligible)  # offset 49 contains the real close

    def test_full_window_cannot_borrow_from_the_next_episode(self):
        rows = [{'observation.state': [0.] * 6 + [1., 1.], 'action': [0.] * 6 + [1.],
                 '_physical_source': '0'} for _ in range(120)]
        episodes = {0: list(range(49)), 1: list(range(49, 120))}
        # Isolate chunk-boundary handling from transition detection: all labels
        # are open, while the already-checked event marks the first close.
        events = [{'episode': 0, 'position': 48, 'index': 48, 'source': '0'}]
        eligible, first_close = conditioned.find_approach_open_anchors(rows, episodes, events)
        self.assertEqual(eligible, {'0': []})
        self.assertEqual(first_close[0]['index'], 48)
        self.assertNotIn(1, first_close)

    def test_balanced_groups_and_rotation_leave_the_base_profile_unchanged(self):
        rows, proof, train, _ = fixture()
        baseline = training.TransitionSampler(rows, train, proof, seed=33)
        comparison = training.TransitionSampler(rows, train, proof, seed=33)
        sampler = ApproachOpenSampler(rows, train, proof, seed=34)
        groups = [sampler.draw_balanced(4) for _ in range(10)]
        fourth = []
        for group in groups:
            categories = [detail['category'] for _, detail in group]
            for category in ('close', 'open', 'motion'):
                self.assertEqual(categories.count(category), 1)
            other = [x for x in categories if x not in ('close', 'open', 'motion')]
            self.assertEqual(len(other), 1)
            fourth.extend(other)
        self.assertEqual(fourth, list(conditioned.APPROACH_OPEN_FOURTH_SCHEDULE) * 2)
        self.assertEqual(training.TransitionSampler.VERSION, 'source-transition-motion-hold-balanced-batch-v2')
        self.assertEqual(training.TransitionSampler.FOURTH_SCHEDULE, ('all', 'hold', 'all', 'hold', 'all'))
        self.assertEqual([baseline.draw_balanced(4) for _ in range(8)],
                         [comparison.draw_balanced(4) for _ in range(8)])
        self.assertNotEqual(sampler.VERSION, baseline.VERSION)
        with self.assertRaisesRegex(ValueError, 'multiple of four'):
            sampler.draw_balanced(3)

    def test_training_factory_uses_the_selected_profile(self):
        rows, proof, train, _ = fixture()
        default = training.make_sampler(training.DEFAULT_SAMPLING_PROFILE, rows, train, proof, seed=9)
        optional = training.make_sampler(training.APPROACH_SAMPLING_PROFILE, rows, train, proof, seed=9)
        self.assertEqual(default.VERSION, training.TransitionSampler.VERSION)
        self.assertEqual(optional.VERSION, conditioned.APPROACH_OPEN_VERSION)
        self.assertEqual([detail['category'] for _, detail in optional.draw_balanced(4)].count('approach_open'), 1)
        self.assertEqual(training.resolve_sampling_profile(None), training.DEFAULT_SAMPLING_PROFILE)

    def test_source_uniformity_and_per_source_coverage_report(self):
        rows, proof, train, _ = fixture()
        sampler = ApproachOpenSampler(rows, train, proof, seed=20)
        self.assertEqual({k: len(v) for k, v in sampler.approach_open_by_source.items()},
                         {'0': 72, '1': 151, '3': 0})
        sampled = [sampler.draw_category('approach_open') for _ in range(6000)]
        counts = {source: sum(detail['source'] == source for _, detail in sampled) for source in ('0', '1')}
        self.assertTrue(.47 < counts['0'] / len(sampled) < .53, counts)
        self.assertTrue(all(index in sampler.approach_open_by_source[detail['source']]
                            for index, detail in sampled))
        report = sampler.report()['approach_open']
        self.assertEqual(report['draws'], 6000)
        self.assertEqual(report['eligible'], 223)
        for source in ('0', '1'):
            indices = {index for index, detail in sampled if detail['source'] == source}
            self.assertEqual(report['per_source'][source],
                             {'eligible': len(sampler.approach_open_by_source[source]),
                              'draws': counts[source], 'sampled_unique': len(indices)})
            self.assertEqual(len(indices), len(sampler.approach_open_by_source[source]))
        self.assertEqual(report['per_source']['3'], {'eligible': 0, 'draws': 0, 'sampled_unique': 0})

    def test_resume_restores_all_random_queues_counts_and_unique_indices(self):
        rows, proof, train, _ = fixture()
        original = ApproachOpenSampler(rows, train, proof, seed=19)
        for _ in range(13):
            original.draw()
        original.draw_balanced(12)
        original.draw_category('approach_open')
        state = pickle.loads(pickle.dumps(original.getstate()))
        restored = ApproachOpenSampler(rows, train, proof, seed=44)
        restored.setstate(state)
        self.assertEqual(original.report(), restored.report())
        for _ in range(30):
            self.assertEqual(original.draw(), restored.draw())
            self.assertEqual(original.draw_balanced(8), restored.draw_balanced(8))
            self.assertEqual(original.draw_category('approach_open'), restored.draw_category('approach_open'))
        self.assertEqual(original.getstate(), restored.getstate())
        self.assertEqual(original.report(), restored.report())

    def test_resume_rejects_other_profiles_and_changed_dataset_identity(self):
        rows, proof, train, _ = fixture()
        original = ApproachOpenSampler(rows, train, proof, seed=1)
        original.draw_balanced(4)
        old = training.TransitionSampler(rows, train, proof, seed=1)
        with self.assertRaisesRegex(ValueError, 'different sampling strategy'):
            original.setstate(old.getstate())
        with self.assertRaisesRegex(ValueError, 'different sampling strategy'):
            old.setstate(original.getstate())
        changed = copy.deepcopy(rows)
        changed[train[0][0]]['observation.state'][0] += .001
        changed_sampler = ApproachOpenSampler(changed, train, proof, seed=1)
        self.assertEqual(original.events, changed_sampler.events)
        with self.assertRaisesRegex(ValueError, 'identity'):
            changed_sampler.setstate(original.getstate())
        bad_state = original.getstate()
        bad_state['approach_open_seen_by_source']['0'] = [train[0][100]]
        with self.assertRaisesRegex(ValueError, 'Invalid approach-open'):
            original.setstate(bad_state)

    def test_no_eligible_source_and_heldout_episodes_fail_closed(self):
        rows, proof, train, validation = fixture()
        for indices in train.values():
            for index in indices:
                rows[index]['observation.state'][7] = 0.
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            ApproachOpenSampler(rows, train, proof, seed=0)
        with self.assertRaisesRegex(ValueError, 'training episodes only'):
            ApproachOpenSampler(rows, {**train, **validation}, proof, seed=0)


class EarlySwitchContextSamplingTests(unittest.TestCase):
    def test_original_labels_event_offsets_and_training_sources(self):
        rows, proof, train, validation = fixture()
        actions = [list(row['action']) for row in rows]
        sampler = EarlySwitchContextSampler(rows, train, proof, seed=4)
        self.assertEqual(sampler.VERSION, conditioned.EARLY_SWITCH_CONTEXT_VERSION)
        for category, entries in sampler.early_switch_events.items():
            kind = category.removeprefix('before_')
            self.assertTrue(entries)
            for entry in entries:
                event = entry['event']
                self.assertIn(event, sampler.events[kind])
                indices = train[event['episode']]
                for offset, index in entry['anchors']:
                    self.assertIn(offset, (-30, -24, -18, -12))
                    pos = event['position'] + offset
                    self.assertEqual(index, indices[pos])
                    self.assertGreaterEqual(pos, 10)
                    self.assertTrue(all(rows[i]['action'][6] == event['old']
                                        for i in indices[pos - 10:pos + 10]))
                    self.assertNotIn(index, validation[5])
        close = next(e for e in sampler.early_switch_events['before_close']
                     if e['event']['episode'] == 0)
        opened = next(e for e in sampler.early_switch_events['before_open']
                      if e['event']['episode'] == 0)
        self.assertEqual([offset for offset, _ in close['anchors']], [-30, -24, -18, -12])
        self.assertEqual([offset for offset, _ in opened['anchors']], [-18, -12])
        self.assertEqual([row['action'] for row in rows], actions)
        with self.assertRaisesRegex(ValueError, 'training episodes only'):
            EarlySwitchContextSampler(rows, {**train, **validation}, proof, seed=4)

    def test_preceding_and_following_targets_must_match_original_opening(self):
        rows, proof, train, _ = fixture()
        sampler = EarlySwitchContextSampler(rows, train, proof, seed=4)
        event = next(e for e in sampler.events['close'] if e['episode'] == 0)
        events = {'close': [event], 'open': []}
        for changed_pos in (69, 70):
            changed = copy.deepcopy(rows)
            changed[train[0][changed_pos]]['action'][6] = .76
            eligible = conditioned.find_early_switch_context_anchors(changed, train, events)
            offsets = [offset for offset, _ in eligible['before_close'][0]['anchors']]
            self.assertNotIn(-30, offsets)
            self.assertIn(-12, offsets)

    def test_fourth_slot_alternates_and_other_slots_are_preserved(self):
        rows, proof, train, _ = fixture()
        sampler = EarlySwitchContextSampler(rows, train, proof, seed=5)
        fourth = []
        for _ in range(20):
            group = sampler.draw_balanced(4)
            categories = [detail['category'] for _, detail in group]
            for kind in ('close', 'open', 'motion'):
                self.assertEqual(categories.count(kind), 1)
            fourth.extend(c for c in categories if c not in ('close', 'open', 'motion'))
        self.assertEqual(fourth, list(conditioned.EARLY_SWITCH_CONTEXT_FOURTH_SCHEDULE) * 2)
        self.assertEqual([c for c in fourth if c.startswith('before_')],
                         ['before_close', 'before_open'] * 2)
        self.assertEqual(conditioned.APPROACH_OPEN_FOURTH_SCHEDULE,
                         ('approach_open', 'all', 'hold', 'approach_open', 'all'))

    def test_every_eligible_event_is_visited_once_per_shuffled_cycle(self):
        rows, proof, train, _ = fixture()
        sampler = EarlySwitchContextSampler(rows, train, proof, seed=12)
        for category, entries in sampler.early_switch_events.items():
            expected = {entry['event']['index'] for entry in entries}
            eligible = {entry['event']['index']: set(entry['anchors']) for entry in entries}
            for _ in range(3):
                seen = set()
                for _ in entries:
                    index, detail = sampler.draw_category(category)
                    seen.add(detail['event_index'])
                    self.assertIn((detail['offset_frames'], index), eligible[detail['event_index']])
                self.assertEqual(seen, expected)

    def test_resume_preserves_pending_event_queues_and_random_draws(self):
        rows, proof, train, _ = fixture()
        original = EarlySwitchContextSampler(rows, train, proof, seed=21)
        original.draw_balanced(28)
        original.draw_category('before_close')
        original.draw_category('before_open')
        state = pickle.loads(pickle.dumps(original.getstate()))
        restored = EarlySwitchContextSampler(rows, train, proof, seed=99)
        restored.setstate(state)
        self.assertEqual(original.report(), restored.report())
        for _ in range(30):
            self.assertEqual(original.draw_balanced(8), restored.draw_balanced(8))
            self.assertEqual(original.draw_category('before_close'), restored.draw_category('before_close'))
            self.assertEqual(original.draw_category('before_open'), restored.draw_category('before_open'))
        self.assertEqual(original.getstate(), restored.getstate())
        old = ApproachOpenSampler(rows, train, proof, seed=1)
        with self.assertRaisesRegex(ValueError, 'different sampling strategy'):
            restored.setstate(old.getstate())

    def test_missing_eligible_switch_direction_fails(self):
        rows, proof, train, _ = fixture()
        # Episode 2 is closed only for ten frames, leaving no old-closed context.
        with self.assertRaisesRegex(ValueError, 'eligible close and open'):
            EarlySwitchContextSampler(rows, {2: train[2]}, proof, seed=0)


if __name__ == '__main__':
    unittest.main()
