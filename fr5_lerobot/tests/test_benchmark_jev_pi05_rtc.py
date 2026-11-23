"""Nominal prefix alignment for the receive-only RTC benchmark."""
import copy
import unittest

from fr5_lerobot.diagnostics.benchmark_jev_pi05_rtc import prior_prefix, summarize_timings


def captured(identifier, stamp):
    return {
        'request': {'id': identifier, 'source_time': stamp},
        'reply': {'actions': [[i * .01] * 6 + [.756] for i in range(50)]},
    }


class RTCBenchmarkTests(unittest.TestCase):
    def test_same_real_predictions_are_shifted_and_only_device_gripper_projected(self):
        previous, current = captured('old', 100.), captured('new', 100.5)
        original = copy.deepcopy(previous)
        window = prior_prefix(previous, current)
        self.assertEqual(window['nominal_past_steps'], 3)
        self.assertEqual(window['retained_steps'], 7)
        self.assertEqual(window['absolute_actions'][0][:6], [.03] * 6)
        self.assertEqual(window['absolute_actions'][-1][:6], [.09] * 6)
        self.assertEqual([row[6] for row in window['absolute_actions']], [.76] * 7)
        self.assertAlmostEqual(window['first_retained_time_minus_current_source_s'], 1 / 6)
        self.assertEqual(window['commitment'], 'hypothetical_prior_prediction_not_executed')
        self.assertEqual(previous, original)

    def test_actual_frame_gap_can_exhaust_prefix_without_fabricated_padding(self):
        window = prior_prefix(captured('old', 100.), captured('new', 102.))
        self.assertEqual(window['retained_steps'], 0)
        self.assertEqual(window['absolute_actions'], [])
        self.assertIsNone(window['first_retained_time_minus_current_source_s'])

    def test_non_grid_source_interval_keeps_explicit_alignment_offset(self):
        window = prior_prefix(captured('old', 100.), captured('new', 100.55))
        self.assertEqual(window['nominal_past_steps'], 3)
        self.assertAlmostEqual(window['first_retained_time_minus_current_source_s'], 4 / 6 - .55)
        with self.assertRaises(ValueError):
            prior_prefix(captured('old', 100.), captured('new', 100.))

    def test_timing_summary_does_not_invent_missing_observations(self):
        self.assertEqual(summarize_timings([]), {})
        rows = [{'timings': {'sampling_s': n, 'postprocess_s': .02,
                             'local_total_s': n + .02, 'peak_allocated_mib': 12000}}
                for n in (.4, .5)]
        summary = summarize_timings(rows)
        self.assertEqual(summary['sampling_s']['count'], 2)
        self.assertAlmostEqual(summary['sampling_s']['median'], .45)
        self.assertEqual(summary['peak_allocated_mib']['max'], 12000)


if __name__ == '__main__':
    unittest.main()
