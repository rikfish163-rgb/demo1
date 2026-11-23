"""Supported-contact settling tests: synthetic telemetry and a virtual clock only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from . import teacher_targets
from .test_teacher_targets import FakeRobot, scope


class VirtualClock:
    def __init__(self):
        self.now = 100.
        self.sleeps = []

    def monotonic(self):
        return self.now

    def time(self):
        return 2000.+self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now = round(self.now+duration, 9)


class ContactSettlingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp', prefix='fr5-contact-settling-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = VirtualClock()
        self.addCleanup(patch.stopall)
        patch.object(teacher_targets, 'time', self.clock).start()
        patch('socket.create_connection', side_effect=AssertionError('network forbidden')).start()
        patch('socket.socket.connect', side_effect=AssertionError('network forbidden')).start()
        self.base = deepcopy(FakeRobot().s)
        self.base.update(gripper_position=100, gripper_motiondone=1)
        self.contact_time = None
        self.reads = 0
        self.profile = lambda elapsed, index: {'gripper_position':77, 'gripper_motiondone':2}
        robot = SimpleNamespace(stream=SimpleNamespace(latest=self.latest))
        self.rpc = SimpleNamespace(call=Mock(side_effect=AssertionError('RPC forbidden')))
        configuration = scope()
        configuration['reviewed_at_monotonic'] = self.clock.now
        (self.root/'scope.json').write_text(json.dumps(configuration))
        self.session = teacher_targets.TeacherSession(self.root, robot=robot, rpc=self.rpc)
        self.session.phase = 'close'
        self.session.append([*self.base['joints'],100.], 'hold', False)
        self.session.send = Mock(side_effect=self.send)

    def latest(self, **kwargs):
        self.assertEqual(kwargs, {'max_age_s':.25})
        self.reads += 1
        state = deepcopy(self.base)
        if self.contact_time is not None:
            state.update(self.profile(round(self.clock.now-self.contact_time, 9), self.reads))
        state.update(frame_counter=self.reads, received_monotonic=self.clock.now,
                     timestamp=self.clock.time())
        return state

    def send(self, method, args, target, *, kind):
        self.assertIsNone(self.contact_time, 'No additional closing command after contact')
        self.assertEqual((method,kind), ('MoveGripper','gripper'))
        self.assertEqual(target[6], args[1])
        self.session.append(target, kind, True)
        self.base.update(gripper_position=args[1], gripper_motiondone=1)
        if args[1] == 76:
            self.base.update(gripper_position=78, gripper_motiondone=2)
            self.contact_time = self.clock.now
        return deepcopy(self.base)

    def report(self):
        return json.loads((self.root/'hold-verification.json').read_text())

    def assert_no_extra_dispatch(self):
        self.assertEqual([call.args[0] for call in self.session.send.call_args_list], ['MoveGripper']*3)
        self.assertEqual([call.args[1][1] for call in self.session.send.call_args_list], [90,80,76])
        self.assertEqual(len(self.session.journal), 4)  # One explicit hold and three closing targets.
        self.assertEqual(self.session.peak_height, 0.)
        self.rpc.call.assert_not_called()

    def test_78_then_76_77_settles_for_three_seconds_and_uses_median(self):
        self.profile = lambda elapsed,index: {
            'gripper_position':78 if elapsed < .1 else (76 if index % 2 else 77),
            'gripper_motiondone':2,
        }
        result = self.session.close_gripper()
        report = self.report()
        self.assertEqual(report['status'], 'stable_status2')
        self.assertEqual(report['initial_contact_position'], 78)
        self.assertGreaterEqual(report['stable_window_s'], 3.)
        self.assertGreaterEqual(self.clock.now-self.contact_time, 3.1-1e-8)
        self.assertLess(self.clock.now-self.contact_time, 6.)
        end = report['samples'][-1]['received_monotonic']
        stable = [s['gripper_position'] for s in report['samples']
                  if s['received_monotonic'] >= end-report['stable_window_s']-1e-8]
        self.assertLessEqual(max(stable)-min(stable), 1.)
        self.assertEqual(report['settled_reference_position'], float(np.median(stable)))
        self.assertEqual(self.session.holding_position, float(np.median(stable)))
        self.assertEqual(result['gripper_motiondone'], 2)
        self.assertTrue(report['reference_acquired_at_support_before_any_lift'])
        self.assert_no_extra_dispatch()

    def test_invalid_feedback_resets_window_before_later_stability(self):
        self.profile = lambda elapsed,index: ({'gripper_position':0,'gripper_motiondone':0}
                                             if 2.4 <= elapsed < 2.5 else
                                             {'gripper_position':77,'gripper_motiondone':2})
        self.session.close_gripper()
        self.assertGreaterEqual(self.clock.now-self.contact_time, 5.5)
        self.assertLess(self.clock.now-self.contact_time, 6.)
        self.assertGreaterEqual(self.report()['stable_window_s'], 3.)
        self.assert_no_extra_dispatch()

    def test_indefinitely_invalid_feedback_fails_within_six_seconds(self):
        self.profile = lambda elapsed,index: {'gripper_position':0, 'gripper_motiondone':0}
        with self.assertRaisesRegex(RuntimeError, 'within6seconds'):
            self.session.close_gripper()
        self.assertGreaterEqual(self.clock.now-self.contact_time, 6.)
        self.assertLess(self.clock.now-self.contact_time, 6.1)
        self.assertIsNone(self.session.holding_position)
        report = self.report()
        self.assertEqual(report['status'], 'failed_before_lift')
        self.assertEqual(report['rejected_or_last_state']['gripper_motiondone'], 0)
        self.assert_no_extra_dispatch()

    def test_indefinitely_unstable_span_never_acquires_reference(self):
        self.profile = lambda elapsed,index: {'gripper_position':76 if index % 2 else 78,
                                             'gripper_motiondone':2}
        with self.assertRaisesRegex(RuntimeError, 'within6seconds'):
            self.session.close_gripper()
        self.assertIsNone(self.session.holding_position)
        self.assertEqual(self.report()['status'], 'failed_before_lift')
        self.assertGreaterEqual(self.clock.now-self.contact_time, 6.)
        self.assert_no_extra_dispatch()

    def test_status3_fails_immediately_and_retains_rejection_state(self):
        self.profile = lambda elapsed,index: {'gripper_position':77,'gripper_motiondone':3}
        with self.assertRaisesRegex(RuntimeError, 'Object loss'):
            self.session.close_gripper()
        self.assertEqual(self.clock.now, self.contact_time)
        self.assertEqual(self.clock.sleeps, [])
        report = self.report()
        self.assertEqual(report['status'], 'failed_before_lift')
        self.assertEqual(report['rejected_or_last_state']['gripper_motiondone'], 3)
        self.assertIsNone(self.session.holding_position)
        self.assert_no_extra_dispatch()

    def test_support_height_drift_is_rejected_before_any_lift(self):
        self.profile = lambda elapsed,index: {'gripper_position':77,'gripper_motiondone':2,
                                             'tcp':[0.,0.,100.06,0.,0.,0.]}
        with self.assertRaisesRegex(RuntimeError, 'original support height'):
            self.session.close_gripper()
        report = self.report()
        self.assertEqual(report['status'], 'failed_before_lift')
        self.assertAlmostEqual(report['rejected_or_last_state']['tcp'][2], 100.06)
        self.assertEqual(self.clock.sleeps, [])
        self.assertIsNone(self.session.holding_position)
        self.assert_no_extra_dispatch()

    def test_held_monitor_remains_strict_after_reference_is_acquired(self):
        self.session.close_gripper()
        self.assertEqual(self.session.holding_position, 77.)
        self.profile = lambda elapsed,index: {'gripper_position':78.01,'gripper_motiondone':2}
        with self.assertRaisesRegex(RuntimeError, 'Stable holding feedback lost'):
            self.session.state(held=True)
        self.assert_no_extra_dispatch()


if __name__ == '__main__':
    unittest.main()
