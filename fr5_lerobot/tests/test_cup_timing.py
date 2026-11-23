"""Causal timing for cup contact, steady lift and return height."""

import unittest

from fr5_jev.cup_timing import CupTiming
from fr5_lerobot.training.stream_observation_history import (
    CUP_ALIGNED_PROFILE, CUP_SETDOWN_PROFILE, CUP_TIMING_PROFILE, action_clock, render_task,
)


class CupTimingTests(unittest.TestCase):
    def setUp(self):
        self.timer = CupTiming({'table_normal': [0., 0., 1.], 'rim_xyz_mm': [0., 0., 110.],
                                'base_xyz_mm': [0., 0., 0.], 'camera_serial': 'gemini'})

    def observe(self, stamp, *, z=95.8, x=159.7, contact=1, rim_z=None, speed=0.,
                motion_done=1, gripper_position=None):
        if gripper_position is None:
            gripper_position = 100 if contact == 0 else 78
        camera = {'source_timestamp_monotonic': stamp, 'source_camera_serial': 'gemini'}
        robot = {'received_monotonic': stamp+.003, 'flange': [x, 71.3, z, 0., 0., 0.],
                 'tool': 0, 'user': 0, 'gripper_position': gripper_position,
                 'tcp_speed': [speed, 0.], 'motion_done': motion_done}
        tracking = {'native_contact': contact, 'rim_minus_jaw_mm': None,
                    'rim_xyz_mm': None if rim_z is None else [0., 0., rim_z]}
        return self.timer.update(camera, robot, tracking)['state']

    def test_contact_and_hold_use_acquisition_time_not_command_or_frame_count(self):
        self.assertEqual(self.observe(100., x=370., contact=0)['phase'], 'approaching')
        self.assertEqual(self.observe(100.17, contact=0)['phase'], 'at_pickup')
        self.assertEqual(self.observe(100.32, contact=None)['phase'], 'unknown')
        self.assertEqual(self.observe(100.48)['phase'], 'grasp_contact')
        self.assertEqual(self.observe(100.67, z=116.)['phase'], 'lifting')
        for t in (100.8, 100.97, 101.15, 101.33):
            state = self.observe(t, z=145.8, rim_z=160.)
        self.assertEqual(state['phase'], 'hover')
        self.assertEqual(state['visual'], 'raised')
        self.assertAlmostEqual(state['hover_seconds'], .53)
        previous = self.observe(101.51, z=145.8)
        self.assertEqual(previous['visual'], 'missing')
        self.assertAlmostEqual(previous['hover_seconds'], .71)
        self.assertEqual(self.observe(101.68, z=139.)['phase'], 'returning')
        self.assertEqual(self.observe(101.87, z=117.)['phase'], 'returning')
        for t in (102.01, 102.19, 102.34, 102.53):
            state = self.observe(t, rim_z=110.)
        self.assertEqual(state['phase'], 'at_table_height')
        self.assertAlmostEqual(state['hover_seconds'], .71)
        self.assertEqual(self.observe(102.7, contact=0, x=172.)['phase'], 'released')
        self.assertEqual(self.observe(102.87, contact=0, x=181.)['phase'], 'released')

    def test_invalid_feedback_preserves_contact_through_close_and_open_gaps(self):
        self.assertEqual(self.observe(1., contact=0)['phase'], 'at_pickup')
        for stamp, position in ((1.2, 99), (1.4, 90), (1.6, 81)):
            state = self.observe(stamp, contact=None, gripper_position=position)
            self.assertEqual(state['phase'], 'unknown')
            self.assertIsNone(self.timer.contact_flange)
            self.assertFalse(self.timer.was_released)
        self.assertEqual(self.observe(1.8, gripper_position=81)['phase'], 'grasp_contact')
        contact_time = self.timer.contact_time
        contact_flange = self.timer.contact_flange.tolist()
        for stamp, position in ((2., 81), (2.2, 90), (2.4, 99)):
            state = self.observe(stamp, contact=None, gripper_position=position)
            self.assertEqual(state['phase'], 'unknown')
            self.assertEqual(self.timer.contact_time, contact_time)
            self.assertEqual(self.timer.contact_flange.tolist(), contact_flange)
            self.assertFalse(self.timer.was_released)
        self.assertEqual(self.observe(2.6, contact=0, gripper_position=99)['phase'], 'released')
        self.assertTrue(self.timer.was_released)
        self.assertIsNone(self.timer.contact_flange)
        self.assertEqual(self.observe(2.8, contact=None, gripper_position=100)['phase'], 'unknown')
        self.assertTrue(self.timer.was_released)
        self.assertEqual(self.observe(3., contact=0)['phase'], 'released')

    def test_first_valid_99_percent_open_is_released_at_pickup(self):
        for mode in ({}, {'aligned': True}, {'setdown': True}):
            with self.subTest(mode=mode):
                self.timer = CupTiming({'table_normal': [0., 0., 1.],
                                        'rim_xyz_mm': [0., 0., 110.],
                                        'base_xyz_mm': [0., 0., 0.], 'camera_serial': 'gemini'}, **mode)
                self.observe(1., gripper_position=81)
                state = self.observe(1.2, contact=0, gripper_position=99)
                self.assertEqual(state['phase'], 'released')
                self.assertTrue(self.timer.was_released)
                self.assertIsNone(self.timer.contact_time)
                self.assertIsNone(self.timer.contact_flange)
                self.assertIsNone(state['hover_seconds'])
                if self.timer.aligned:
                    self.assertIsNone(state['support_gap_mm'])
                if self.timer.setdown:
                    self.assertIsNone(state['setdown_age_s'])
                self.assertEqual(self.observe(1.4, contact=0)['phase'], 'released')

    def test_close_without_previous_contact_does_not_report_release(self):
        for stamp, position in ((1., 100), (1.2, 99), (1.4, 90), (1.6, 81), (1.8, 0)):
            state = self.observe(stamp, contact=0, gripper_position=position)
            self.assertEqual(state['phase'], 'at_pickup')
            self.assertFalse(self.timer.was_released)
            self.assertIsNone(self.timer.contact_time)
        state = self.observe(2., contact=1, gripper_position=81)
        self.assertEqual(state['phase'], 'grasp_contact')
        self.assertEqual(self.timer.contact_time, 2.)
        self.assertFalse(self.timer.was_released)

    def test_retreat_peak_drop_clears_release_for_second_approach(self):
        for mode in ({'aligned': True}, {'setdown': True}):
            with self.subTest(mode=mode):
                self.timer = CupTiming({'table_normal': [0., 0., 1.],
                                        'rim_xyz_mm': [0., 0., 110.],
                                        'base_xyz_mm': [0., 0., 0.], 'camera_serial': 'gemini'}, **mode)
                self.observe(1.)
                self.assertEqual(self.observe(1.2, contact=0, gripper_position=99)['phase'], 'released')
                for stamp, x in ((1.4, 162.7), (1.6, 171.7), (1.8, 203.7), (2., 198.7)):
                    self.assertEqual(self.observe(stamp, x=x, contact=0)['phase'], 'released')
                    self.assertTrue(self.timer.was_released)
                self.assertAlmostEqual(self.timer.retreat_peak, 44.)
                self.assertEqual(self.observe(2.2, x=197.7, contact=0)['phase'], 'approaching')
                self.assertFalse(self.timer.was_released)
                self.assertIsNone(self.timer.retreat_peak)
                self.assertEqual(self.observe(2.4, x=169.7, contact=0)['phase'], 'at_pickup')
                self.assertEqual(self.observe(2.6, contact=None, gripper_position=81)['phase'], 'unknown')
                self.assertEqual(self.observe(2.8, gripper_position=81)['phase'], 'grasp_contact')
                self.assertEqual(self.timer.contact_time, 2.8)
                self.assertFalse(self.timer.was_released)

    def test_far_contact_and_gap_do_not_invent_pickup_or_elapsed_time(self):
        self.assertEqual(self.observe(100., x=400.)['phase'], 'unknown')
        self.assertEqual(self.observe(100.2)['phase'], 'grasp_contact')
        self.assertIsNone(self.observe(100.9, z=145.)['hover_seconds'])
        self.assertEqual(self.observe(101.08, z=145.)['phase'], 'unknown')
        with self.assertRaisesRegex(ValueError, 'ordered'):
            self.observe(101.08)

    def test_rim_remaining_on_table_disagrees_with_raised_tool(self):
        self.observe(1.)
        for t in (1.17, 1.34, 1.51, 1.68):
            state = self.observe(t, z=146., rim_z=110.)
        self.assertEqual(state['phase'], 'unknown')
        self.assertEqual(state['visual'], 'disagrees')

    def test_prompt_contains_measured_seconds_and_missing_rim(self):
        prompt = render_task('抓杯，悬停五秒', [None]*6, CUP_TIMING_PROFILE,
                             initial_cup_uv=[1400., 700.],
                             cup_tracking={'rim_minus_jaw_mm': None, 'native_contact': 1},
                             cup_timing={'phase': 'hover', 'hover_seconds': 4.96, 'visual': 'missing'})
        self.assertIn('Cup state: hover; hover s: 4.5; rim: missing.', prompt)

    def test_action_clock_distinguishes_elapsed_from_requested_and_return(self):
        task = '抓取桌面上的透明杯子，在空中维持5秒后放下并松开夹爪，再次抓起，在空中维持5秒后放下并松开夹爪'
        hover = {'phase': 'hover', 'hover_seconds': 3.75, 'visual': 'missing'}
        self.assertEqual(action_clock(task, hover)[:2], [1.25, 1.])
        self.assertEqual(action_clock(task, {**hover, 'phase': 'returning'})[:2], [0., 0.])
        self.assertEqual(action_clock('抓杯', hover)[:2], [0., 0.])

    def test_aligned_observer_distinguishes_reapproach_and_return_height(self):
        self.timer = CupTiming({'table_normal': [0., 0., 1.], 'rim_xyz_mm': [0., 0., 110.],
                                'base_xyz_mm': [0., 0., 0.], 'camera_serial': 'gemini'}, aligned=True)
        self.observe(1., contact=1)
        for t in (1.2, 1.4, 1.6, 1.8, 2.):
            state = self.observe(t, z=145.8, rim_z=160.)
        self.assertAlmostEqual(state['support_gap_mm'], 50.)
        before = state['hover_seconds']
        state = self.observe(2.2, z=144.2)
        self.assertEqual(state['phase'], 'hover')
        state = self.observe(2.4, z=142.2)
        self.assertEqual(state['phase'], 'returning')
        self.assertAlmostEqual(state['hover_seconds'], before+.2)
        for t in (2.6, 2.8, 3., 3.2):
            state = self.observe(t, z=100.8, rim_z=110.)
        self.assertEqual(state['phase'], 'returning')
        self.assertAlmostEqual(state['support_gap_mm'], 5.)
        for t in (3.4, 3.6, 3.8, 4.):
            state = self.observe(t, rim_z=110.)
        self.assertEqual(state['phase'], 'at_table_height')
        self.observe(4.2, contact=0)
        self.assertEqual(self.observe(4.4, x=203., contact=0)['phase'], 'released')
        reapproach = self.observe(4.6, x=186., contact=0, rim_z=110.)
        self.assertEqual(reapproach['phase'], 'approaching')
        clock = action_clock('在空中维持5秒', reapproach,
                             cup_tracking={'rim_minus_jaw_mm': [-38., 0., 0.], 'native_contact': 0},
                             profile=CUP_ALIGNED_PROFILE)
        self.assertEqual(len(clock), 24)
        self.assertLess(clock[16], -.6)
        self.assertEqual(clock[19:], [1., 0., 1., 0., 0.])

    def test_hover_start_excludes_fast_lift_even_when_near_the_stable_height(self):
        self.timer = CupTiming({'table_normal': [0., 0., 1.], 'rim_xyz_mm': [0., 0., 110.],
                                'base_xyz_mm': [0., 0., 0.], 'camera_serial': 'gemini'}, setdown=True)
        self.observe(1.)
        self.observe(1.2, z=145., speed=11.27, motion_done=0)
        for stamp in (1.4, 1.6, 1.8):
            state = self.observe(stamp, z=145.8, speed=.96, motion_done=0)
        self.assertEqual(state['phase'], 'hover')
        self.assertAlmostEqual(self.timer.hover_start, 1.4)
        self.assertAlmostEqual(state['hover_seconds'], .4)
        state = self.observe(2., z=145.8, speed=.96, motion_done=0)
        self.assertEqual(state['phase'], 'hover')
        self.assertAlmostEqual(self.timer.hover_start, 1.4)
        self.assertAlmostEqual(state['hover_seconds'], .6)

    def test_speed_noise_preserves_an_established_hover_clock(self):
        self.timer = CupTiming({'table_normal': [0., 0., 1.], 'rim_xyz_mm': [0., 0., 110.],
                                'base_xyz_mm': [0., 0., 0.], 'camera_serial': 'gemini'}, setdown=True)
        self.observe(1.)
        for stamp in (1.2, 1.4, 1.6, 1.8, 2.):
            self.observe(stamp, z=145.8, speed=.8)
        start = self.timer.hover_start
        state = self.observe(2.2, z=145.8, speed=11.)
        self.assertEqual(state['phase'], 'hover')
        self.assertEqual(self.timer.hover_start, start)
        self.assertAlmostEqual(state['hover_seconds'], 2.2-start)

    def test_setdown_clock_needs_return_height_contact_and_stillness(self):
        self.timer = CupTiming({'table_normal': [0., 0., 1.], 'rim_xyz_mm': [0., 0., 110.],
                                'base_xyz_mm': [0., 0., 0.], 'camera_serial': 'gemini'}, setdown=True)
        self.assertIsNone(self.observe(1., rim_z=110.)['setdown_age_s'])
        for t in (1.2, 1.4, 1.6, 1.8, 2.0):
            state = self.observe(t, z=145.8, rim_z=160.)
        self.assertEqual(state['phase'], 'hover')
        self.assertIsNone(state['setdown_age_s'])
        self.observe(2.2, z=130., speed=40.)
        self.assertIsNone(self.observe(2.4, z=100.8, rim_z=110.)['setdown_age_s'])
        self.assertIsNone(self.observe(2.6, rim_z=110., speed=8.)['setdown_age_s'])
        self.assertEqual(self.observe(2.8)['setdown_age_s'], 0.)
        self.observe(3.)
        state = self.observe(3.2)
        self.assertAlmostEqual(state['setdown_age_s'], .4)
        clock = action_clock('在空中维持5秒', state,
                             cup_tracking={'rim_minus_jaw_mm': None, 'native_contact': 1},
                             profile=CUP_SETDOWN_PROFILE)
        self.assertEqual(len(clock), 25)
        self.assertGreater(clock[-1], 0.)
        self.assertIsNone(self.observe(3.4, contact=None)['setdown_age_s'])
        self.assertEqual(self.observe(3.6)['setdown_age_s'], 0.)
        self.assertEqual(self.observe(4.)['setdown_age_s'], 0.)


if __name__ == '__main__':
    unittest.main()
