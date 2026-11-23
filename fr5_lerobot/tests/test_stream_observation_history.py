"""CPU tests for selecting actual recorded observation history."""

import unittest

from fr5_lerobot.training.stream_observation_history import (
    CUP_HISTORY_PROFILE,
    CUP_SETDOWN_CLEAN_PROFILE,
    CUP_SETDOWN_FIXED_PROFILE,
    CUP_SETDOWN_PROFILE,
    CUP_TIMING_PHASES,
    CUP_TIMING_VISION,
    HISTORY_OFFSETS_S,
    HISTORY_PROFILE,
    action_clock,
    context_metadata,
    render_task,
    select_observation_history,
)


class ObservationHistoryTests(unittest.TestCase):
    def test_initial_seconds_have_missing_history(self):
        samples = [(i / 6, [i] * 6 + [1.0, 1.0]) for i in range(13)]
        result = select_observation_history(samples, 2.0)
        self.assertEqual(result[0.5]['timestamp'], 1.5)
        self.assertEqual(result[1.0]['state'][:6], [6] * 6)
        self.assertEqual(result[2.0]['timestamp'], 0.0)
        self.assertIsNone(result[4.0])
        self.assertIsNone(result[6.0])
        self.assertIsNone(result[8.0])

    def test_dropped_frames_choose_older_sample_and_report_actual_age(self):
        samples = [(i / 6, [i] * 6 + [0.78, 1.0]) for i in range(61)
                   if i not in (30, 31, 32)]
        result = select_observation_history(samples, 6.0)
        self.assertEqual(result[1.0]['timestamp'], 29 / 6)
        self.assertAlmostEqual(result[1.0]['age_s'], 7 / 6)
        self.assertEqual(result[1.0]['state'][:6], [29] * 6)
        self.assertGreater(result[1.0]['age_s'], 1.0)

    def test_invalid_feedback_is_preserved_without_substitution(self):
        state = [1, 2, 3, 4, 5, 6, 0.0, 0.0]
        samples = [(0.0, [0] * 6 + [0.78, 1.0]), (1.0, state),
                   (2.0, [2] * 6 + [0.78, 1.0])]
        result = select_observation_history(samples, 2.0)
        self.assertEqual(result[1.0]['state'], state)
        result[1.0]['state'][6] = 99
        self.assertEqual(state[6:], [0.0, 0.0])

    def test_never_uses_sample_after_historical_query_or_current_time(self):
        samples = [(0.0, [0] * 8), (1.49, [1] * 8),
                   (1.51, [2] * 8), (2.2, [3] * 8)]
        result = select_observation_history(samples, 2.0)
        self.assertEqual(result[0.5]['timestamp'], 1.49)
        self.assertEqual(result[0.5]['state'], [1] * 8)
        for offset, entry in result.items():
            if entry is not None:
                self.assertLessEqual(entry['timestamp'], 2.0 - offset)

    def test_empty_samples_and_duplicate_timestamp(self):
        self.assertEqual(select_observation_history([], 10),
                         {offset: None for offset in HISTORY_OFFSETS_S})
        result = select_observation_history([(1, [1] * 8), (1, [2] * 8)], 1.5)
        self.assertEqual(result[0.5]['state'], [2] * 8)

    def test_rejects_unordered_samples_and_non_state8(self):
        with self.assertRaisesRegex(ValueError, 'ordered'):
            select_observation_history([(2, [0] * 8), (1, [0] * 8)], 3)
        with self.assertRaisesRegex(ValueError, 'eight'):
            select_observation_history([(0, [0] * 7)], 1)

    def test_render_preserves_invalid_feedback_missing_slots_and_raw_task(self):
        task = '抓取杯子，然后放下'
        state = [1.2345, -.5, 0, 1.1, -2, .001, 0.0, 0.0]
        history = [{'age_s': .53, 'state': state}] + [None] * 5
        actual = render_task(task, history, HISTORY_PROFILE)
        self.assertTrue(actual.startswith(task + ' '))
        self.assertIn('age_s:joint_mrad,grip_pct,valid', actual)
        self.assertIn('0.53:1234,-500,0,1100,-2000,1,0,0', actual)
        self.assertTrue(actual.endswith(';'.join(['-'] * 5)))
        self.assertEqual(state[-2:], [0.0, 0.0])
        self.assertEqual(render_task(task, None), task)
        with self.assertRaisesRegex(ValueError, 'six'):
            render_task(task, None, HISTORY_PROFILE)
        with self.assertRaisesRegex(ValueError, 'age or state'):
            render_task(task, [{'age_s': .49, 'state': state}] + [None] * 5, HISTORY_PROFILE)

    def test_initial_cup_distinguishes_real_bench_positions_and_requires_observation(self):
        history = [None]*6
        old = render_task('抓杯', history, CUP_HISTORY_PROFILE, initial_cup_uv=[1342.51, 700.82])
        current = render_task('抓杯', history, CUP_HISTORY_PROFILE, initial_cup_uv=[1429.53, 696.82])
        self.assertNotEqual(old, current)
        self.assertEqual(old.split(' History')[1], current.split(' History')[1])
        with self.assertRaisesRegex(ValueError, 'initial cup rim center'):
            render_task('抓杯', history, CUP_HISTORY_PROFILE)

    def test_fixed_setdown_matches_original_in_all_25_clock_channels(self):
        task = '抓起杯子，在空中维持5秒，然后放下'
        for phase in CUP_TIMING_PHASES:
            for visual in CUP_TIMING_VISION:
                for rim, contact in (([12., -60., 90.], 1), (None, None), ([0., 0., 0.], 0)):
                    returning = phase in ('returning', 'at_table_height')
                    timing = {'phase': phase, 'hover_seconds': 1.25 if phase == 'hover' else None,
                              'visual': visual, 'support_gap_mm': .75 if returning else None,
                              'setdown_age_s': .15 if returning else None}
                    tracking = {'rim_minus_jaw_mm': rim, 'native_contact': contact}
                    with self.subTest(phase=phase, visual=visual, contact=contact):
                        original = action_clock(task, timing, cup_tracking=tracking, profile=CUP_SETDOWN_PROFILE)
                        fixed = action_clock(task, timing, cup_tracking=tracking, profile=CUP_SETDOWN_FIXED_PROFILE)
                        self.assertEqual(len(fixed), 25)
                        self.assertEqual(fixed, original)

    def test_fixed_setdown_keeps_original_age_offset_and_clean_keeps_its_scale(self):
        tracking = {'rim_minus_jaw_mm': None, 'native_contact': 1}
        for age, original_age, clean_age in ((None, 0., 0.), (0., .075, 0.), (.15, .15, .5),
                                              (.30, .225, 1.), (.60, .375, 1.), (1.85, 1., 1.), (2., 1., 1.)):
            timing = {'phase': 'returning', 'hover_seconds': None, 'visual': 'missing',
                      'support_gap_mm': 0., 'setdown_age_s': age}
            with self.subTest(age=age):
                original = action_clock('放杯', timing, cup_tracking=tracking, profile=CUP_SETDOWN_PROFILE)
                fixed = action_clock('放杯', timing, cup_tracking=tracking, profile=CUP_SETDOWN_FIXED_PROFILE)
                clean = action_clock('放杯', timing, cup_tracking=tracking, profile=CUP_SETDOWN_CLEAN_PROFILE)
                self.assertEqual(fixed, original)
                self.assertEqual(clean[:-1], original[:-1])
                self.assertAlmostEqual(fixed[-1], original_age)
                self.assertAlmostEqual(clean[-1], clean_age)

    def test_fixed_setdown_renders_exactly_the_original_task(self):
        task = '抓起杯子，在空中维持5秒，然后放下'
        history = [{'age_s': .63, 'state': [.01, -.02, .1, 1.617, -.02, -1.17, 0., 0.]},
                   {'age_s': 1.13, 'state': [.02, -.03, .1, 1.618, -.01, -1.16, .76, 1.]}] + [None] * 4
        for timing, tracking in (
                ({'phase': 'hover', 'hover_seconds': 2.75, 'visual': 'raised',
                  'support_gap_mm': 50., 'setdown_age_s': None},
                 {'rim_minus_jaw_mm': [12., -5., 3.], 'native_contact': 1}),
                ({'phase': 'returning', 'hover_seconds': None, 'visual': 'missing',
                  'support_gap_mm': 0., 'setdown_age_s': .15},
                 {'rim_minus_jaw_mm': None, 'native_contact': 1})):
            with self.subTest(phase=timing['phase']):
                values = dict(initial_cup_uv=[1429.53, 696.82], cup_tracking=tracking, cup_timing=timing)
                original = render_task(task, history, CUP_SETDOWN_PROFILE, **values)
                fixed = render_task(task, history, CUP_SETDOWN_FIXED_PROFILE, **values)
                self.assertEqual(fixed, original)
                self.assertIn('tcp_xyz_mm_from_cup,rpy_mrad', fixed)

    def test_fixed_setdown_metadata_changes_only_profile_and_input_description(self):
        original = context_metadata(CUP_SETDOWN_PROFILE)
        fixed = context_metadata(CUP_SETDOWN_FIXED_PROFILE)
        clean = context_metadata(CUP_SETDOWN_CLEAN_PROFILE)
        self.assertEqual(fixed['cup_timing']['setdown']['scale_s'], 2.)
        self.assertEqual(clean['cup_timing']['setdown']['scale_s'], .30)
        self.assertEqual(fixed['action_clock']['input'], clean['action_clock']['input'])
        self.assertNotEqual(fixed['action_clock']['input'], original['action_clock']['input'])
        fixed['profile'] = original['profile']
        fixed['action_clock']['input'] = original['action_clock']['input']
        self.assertEqual(fixed, original)


if __name__ == '__main__':
    unittest.main()
