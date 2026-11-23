"""Offline profile and execution tests; all controller I/O is mocked."""
from copy import deepcopy
import hashlib
import io
import json
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from .prepare_varied_session import build_manual_action_plan, build_routes, finalize, main, prepare
from .teacher_targets import TeacherSession
from .teacher_varied import (CONTINUOUS_PROFILE, LEGACY_PROFILE, PROFILE_SPEEDS,
                             STAGES, VariedSession, validate_varied_scope)
from .test_teacher_targets import FakeRobot, FakeRPC, FakeStorage


def pose(z):
    return [0., 0., float(z), 0., 0., 0.]


def joints(z):
    return [0., 0., (float(z) - 100.) / 100., 0., 90., 0.]


class SyntheticIK:
    def GetInverseKinRef(self, mode, target, reference):
        return [0, *joints(target[2])]

    def GetForwardKin(self, target):
        return [0, *pose(100. + target[2] * 100.)]


def routes(crossing_return=False):
    return {
        'approach_clear': [pose(300), pose(220), pose(200)],
        'approach_near': [pose(200), pose(140)],
        'approach_contact': [pose(140), pose(120)],
        'close': [pose(120), pose(120)],
        'lift_probe': [pose(120), pose(130)],
        'lift_full': [pose(130), pose(170)],
        'return': ([pose(170), pose(120)] if crossing_return else
                   [pose(170), pose(130), pose(122), pose(120)]),
        'release': [pose(120), pose(120)],
        'retreat_probe': [pose(120), pose(130)],
        'retreat_clear': [pose(130), pose(200)],
        'retreat_high': [pose(200), pose(220), pose(300)],
    }


def make_plan(route, profile=LEGACY_PROFILE):
    rows = []
    for phase, points in route.items():
        for segment, (start, end) in enumerate(zip(points, points[1:])):
            for fraction, point in ((0., start), (1., end)):
                rows.append({'phase': phase, 'segment': segment, 'fraction': fraction,
                             'q': joints(point[2])})
    with patch('fr5_jev.prepare_varied_session.controller', return_value=SyntheticIK()):
        return build_manual_action_plan(route, rows, joints(300), profile=profile)


def distance(action):
    return float(np.linalg.norm(np.array(action['source_pose'][:3]) -
                                np.array(action['wire']['desc_pos'][:3])))


def write_reviewed_geometry(root, route, plan, *, old_legacy=False):
    plan = deepcopy(plan)
    if old_legacy:
        plan.pop('profile')
    action_raw = json.dumps(plan).encode()
    (root / 'action-plan.json').write_bytes(action_raw)
    action_hash = hashlib.sha256(action_raw).hexdigest()
    geometry = {
        'status': 'prepared_for_visual_review', 'created_unix': time.time(),
        'initial_tcp': route['approach_clear'][0],
        'route_model': {'contact_pose': route['close'][0]},
        'routes': route, 'action_plan': plan, 'action_plan_sha256': action_hash,
        'kinematics': {'minimum_joint_margin_deg': 30, 'max_fk_error_mm': 0.,
                       'max_fk_rotation_error_deg': 0.},
        'cup_center_change_mm': 0.,
        'expanded_tool_min_table_clearances_mm': {'tool': 10.},
        'cup_tool_collision_samples': {
            'summary': {'finger_sample_overlap_count': 0},
            'approach_sweep_diagnostic': {'pose_count': 1, 'bodies': {
                name: {'poses_with_overlap': 0} for name in ('-1', '1', 'housing')}},
        },
    }
    if not old_legacy:
        geometry['profile'] = plan['profile']
    raw = json.dumps(geometry).encode()
    (root / 'geometry.json').write_bytes(raw)
    review = {'geometry_sha256': hashlib.sha256(raw).hexdigest(),
              'action_plan_sha256': action_hash, 'reviewed_at_unix': time.time(),
              'tool_geometry_independently_verified': True,
              'exterior_finger_sweep_verified': True, 'visible_path_reviewed': True,
              'reviewer': 'synthetic', 'evidence': ['synthetic']}
    (root / 'scene-review.json').write_text(json.dumps(review))
    return geometry, plan


class ContinuousProfileTests(unittest.TestCase):
    def test_continuous_route_preserves_measured_attitude(self):
        start=np.array([150.,-594.,169.,91.79,-2.95,-65.75])
        route,model=build_routes(start,np.array([-230.,-656.,-4.]),np.array([0.,0.,1.]),
                                 profile=CONTINUOUS_PROFILE)
        for points in route.values():
            for point in points:np.testing.assert_allclose(point[3:],start[3:])
        self.assertEqual(model['orientation_policy'],'preserve_measured')
    def test_observed_grasp_offset_proposes_new_pose_without_replaying_joints(self):
        from fr5_runtime.kinematics import transform
        start=np.array([150.,-594.,169.,91.79,-2.95,-65.75])
        base=np.array([-230.,-656.,-4.]);normal=np.array([0.,0.,1.])
        reference={'effective_grasp_offset_flange_mm':[3.,4.,170.]}
        route,model=build_routes(start,base,normal,profile=CONTINUOUS_PROFILE,grasp_reference=reference)
        contact=np.array(model['contact_pose'])
        np.testing.assert_allclose(contact[:3]+transform(contact)[:3,:3]@np.array([3.,4.,170.]),base+87*normal)
        with self.assertRaises(ValueError):
            build_routes(start,base,normal,profile=CONTINUOUS_PROFILE,prealigned=True,grasp_reference=reference)
        with self.assertRaises(ValueError):
            build_routes(start,base,normal,profile=CONTINUOUS_PROFILE,
                         grasp_reference={'effective_grasp_offset_flange_mm':[3.,4.,500.]})

    def test_supervised_nominal_basis_is_explicit_and_requires_visual_workflow(self):
        write_reviewed_geometry(self.root,self.route,self.continuous)
        p=self.root/'scene-review.json';review=json.loads(p.read_text())
        review.update(verification_basis='operator_supervised_nominal',
                      tool_geometry_independently_verified=False,exterior_finger_sweep_verified=False,
                      nominal_tool_dimensions_reviewed=True,visible_empty_approach_reviewed=True,
                      onsite_clearance_source='SYNTHETIC operator supervision statement')
        p.write_text(json.dumps(review));finalize(self.root)
        scope=json.loads((self.root/'scope.json').read_text())
        self.assertFalse(scope['exterior_contact_route_reviewed'])
        self.assertFalse(scope['visible_local_sweep_clear'])
        self.assertTrue(scope['requires_runtime_visual_alignment'])
        validate_varied_scope(scope)
        with self.assertRaisesRegex(RuntimeError,'continuous visual workflow'):
            VariedSession(self.root,robot=FakeRobot(),rpc=Mock())
        review['tool_geometry_independently_verified']=True;p.write_text(json.dumps(review))
        with self.assertRaisesRegex(RuntimeError,'explicit assumptions'):finalize(self.root)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.route = routes()
        self.legacy = make_plan(self.route)
        self.continuous = make_plan(self.route, CONTINUOUS_PROFILE)

    def tearDown(self):
        self.tmp.cleanup()

    def finalized_scope(self, plan=None, *, old_legacy=False):
        plan = self.continuous if plan is None else plan
        write_reviewed_geometry(self.root, self.route, plan, old_legacy=old_legacy)
        finalize(self.root)
        return json.loads((self.root / 'scope.json').read_text())

    def session(self):
        self.finalized_scope()
        robot = FakeRobot()
        robot.s.update(tcp=pose(300), flange=pose(300), joints=joints(300),
                       gripper_position=100, gripper_motiondone=1)
        rpc = FakeRPC(robot)
        session = VariedSession(self.root, robot=robot, rpc=rpc)
        session.storage = FakeStorage()
        session.capture = SimpleNamespace(assert_healthy=Mock(), phase=Mock())
        session.limits = np.array([[-180., 180.]] * 6)
        session.append([*joints(300), 100], 'hold', False)
        return session, robot, rpc

    def test_same_route_has_fewer_actions_and_exact_larger_free_commands(self):
        self.assertEqual(self.legacy['profile'], LEGACY_PROFILE)
        self.assertEqual(self.continuous['profile'], CONTINUOUS_PROFILE)
        self.assertLess(len(self.continuous['actions']), len(self.legacy['actions']))
        free = {'approach_clear', 'approach_near', 'retreat_clear', 'retreat_high'}
        new_moves = [a for a in self.continuous['actions'] if a['kind'] == 'MoveL']
        old_moves = [a for a in self.legacy['actions'] if a['kind'] == 'MoveL']
        self.assertTrue(any(distance(a) > 15.05 for a in new_moves if a['phase'] in free))
        self.assertTrue(all(distance(a) <= 15. + 1e-8 for a in old_moves))
        for phase in free:
            phase_moves = [a for a in new_moves if a['phase'] == phase]
            self.assertTrue(phase_moves)
            self.assertTrue(all(distance(a) <= 80. + 1e-8 and a['wire']['vel'] == 10.
                                for a in phase_moves))
        for phase in STAGES:
            phase_moves = [a for a in new_moves if a['phase'] == phase]
            for action in phase_moves:
                self.assertEqual(action['target_pose'], action['wire']['desc_pos'])
                self.assertEqual(action['joint_target_deg'], action['wire']['joint_pos'])
            for a, b in zip(phase_moves, phase_moves[1:]):
                self.assertEqual(a['target_pose'], b['source_pose'])
        self.assertEqual([a['wire']['pos'] for a in self.continuous['actions']
                          if a['phase'] == 'close'], [90, 80, 76])

    def test_phase_limits_and_timeout_for_every_planned_move(self):
        session, _, _ = self.session()
        for action in self.continuous['actions']:
            if action['kind'] != 'MoveL':
                continue
            phase = action['phase']
            session.phase = phase
            start, target = action['source_pose'], action['wire']['desc_pos']
            d, vel = distance(action), action['wire']['vel']
            if phase in ('approach_contact', 'lift_probe', 'retreat_probe'):
                self.assertLessEqual(d, 10. + 1e-8)
                self.assertEqual(vel, 1.)
            if phase == 'lift_full' or (phase == 'return' and min(start[2], target[2]) >= 122.):
                self.assertLessEqual(d, 40. + 1e-8)
                self.assertEqual(vel, 3.)
            if phase == 'return' and min(start[2], target[2]) < 122.:
                self.assertLessEqual(d, 1. + 1e-8)
                self.assertEqual(vel, 1.)
            rpc_timeout, completion_timeout = session.command_timeouts('arm', {'tcp': start}, target)
            self.assertLessEqual(rpc_timeout, 35.1)
            self.assertEqual(completion_timeout, rpc_timeout + 3.)

    def test_return_crossing_is_split_exactly_at_final_two_mm(self):
        plan = make_plan(routes(crossing_return=True), CONTINUOUS_PROFILE)
        returns = [a for a in plan['actions'] if a['phase'] == 'return']
        self.assertTrue(any(a['target_pose'][2] == 122. for a in returns))
        for action in returns:
            start, end = action['source_pose'][2], action['target_pose'][2]
            if end >= 122. - 1e-8:
                self.assertLessEqual(distance(action), 40. + 1e-8)
                self.assertEqual(action['wire']['vel'], 3.)
            else:
                self.assertLessEqual(start, 122. + 1e-8)
                self.assertLessEqual(distance(action), 1. + 1e-8)
                self.assertEqual(action['wire']['vel'], 1.)

    def test_runtime_samples_large_move_at_one_mm_and_records_wire_target(self):
        session, _, rpc = self.session()
        session.phase = 'approach_clear'
        action = next(a for a in session.planned_actions if a['phase'] == session.phase)
        self.assertEqual(distance(action), 80.)
        result = session.move_to(action)
        ik_calls = [call for call in rpc.calls if call[0] == 'GetInverseKinRef']
        self.assertEqual(len(ik_calls), 80)
        checks = json.loads((self.root / 'path-0002.json').read_text())['checks']
        self.assertEqual(len(checks), 81)
        self.assertTrue(all(abs(b['pose'][2] - a['pose'][2]) <= 1. + 1e-8
                            for a, b in zip(checks, checks[1:])))
        move = next(call for call in rpc.calls if call[0] == 'MoveL')
        self.assertEqual(move[1][1], action['wire']['desc_pos'])
        self.assertEqual(move[1][4], 10.)
        self.assertEqual(result['tcp'], action['target_pose'])
        self.assertEqual(session.journal[-1]['target'][:6], move[1][0])

    def test_runtime_rejects_each_oversized_phase_before_ik_or_send(self):
        session, _, rpc = self.session()
        cases = [('approach_clear', 300., 219.99), ('retreat_clear', 120., 200.01),
                 ('lift_full', 130., 170.01), ('return', 165., 124.99),
                 ('approach_contact', 140., 129.99), ('lift_probe', 120., 130.01),
                 ('retreat_probe', 120., 130.01), ('return', 122., 120.99)]
        for phase, source, target in cases:
            with self.subTest(phase=phase):
                action = deepcopy(next(a for a in self.continuous['actions']
                                       if a['phase'] == phase and a['kind'] == 'MoveL'))
                action['source_pose'] = pose(source)
                action['target_pose'] = action['wire']['desc_pos'] = pose(target)
                session.phase, session.phase_action_index = phase, 0
                session.planned_actions = [action]
                session.state = Mock(return_value={'tcp': pose(source), 'joints': joints(source)})
                with self.assertRaisesRegex(ValueError, 'distance|phase limit'):
                    session.move_to(action)
        self.assertFalse(rpc.calls)

    def test_excessive_time_budget_is_rejected(self):
        session, _, _ = self.session()
        session.phase = 'approach_contact'
        with self.assertRaisesRegex(ValueError, 'time budget'):
            session.command_timeouts('arm', {'tcp': pose(150.)}, pose(134.9))

    def test_live_distance_allows_completion_noise_but_not_an_oversized_plan(self):
        session, _, rpc = self.session()
        action=deepcopy(next(a for a in self.continuous['actions'] if a['phase']=='approach_contact'))
        session.phase='approach_contact';session.phase_action_index=0
        session.planned_actions=[action]
        source=action['source_pose'][2]
        session.state=Mock(return_value={'tcp':pose(source+.01),'joints':joints(source+.01)})
        session.send=Mock(return_value={'tcp':action['wire']['desc_pos']})
        session.move_to(action)
        session.send.assert_called_once()
        session.state=Mock(return_value={'tcp':pose(source+.06),'joints':joints(source+.06)})
        with self.assertRaisesRegex(ValueError,'distance limit'):session.move_to(action)

    def test_scope_declares_and_validates_profile_and_speeds(self):
        scope = self.finalized_scope()
        self.assertEqual(scope['profile'], CONTINUOUS_PROFILE)
        self.assertEqual(scope['motion_speed_profile'], PROFILE_SPEEDS[CONTINUOUS_PROFILE])
        self.assertEqual(scope['move_speed'], 10.)
        self.assertEqual(validate_varied_scope(scope)['profile'], CONTINUOUS_PROFILE)
        bad = deepcopy(scope)
        bad['motion_speed_profile']['empty_free'] = 5.
        with self.assertRaisesRegex(ValueError, 'scope'):
            validate_varied_scope(bad)
        bad = deepcopy(scope)
        bad['profile'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            validate_varied_scope(bad)

    def test_mismatched_plan_and_geometry_profiles_are_rejected(self):
        for component in ('plan', 'geometry', 'missing_plan_profile', 'unknown_plan'):
            with self.subTest(component=component):
                scope = self.finalized_scope()
                modified = deepcopy(scope)
                if component == 'geometry':
                    path = self.root / 'geometry.json'
                    obj = json.loads(path.read_text())
                    obj['profile'] = LEGACY_PROFILE
                    raw = json.dumps(obj).encode()
                    path.write_bytes(raw)
                    modified['geometry_sha256'] = hashlib.sha256(raw).hexdigest()
                else:
                    path = self.root / 'action-plan.json'
                    obj = deepcopy(scope['manual_action_plan'])
                    if component == 'plan':
                        obj['profile'] = LEGACY_PROFILE
                    elif component == 'unknown_plan':
                        obj['profile'] = 'unknown'
                    else:
                        obj.pop('profile')
                    raw = json.dumps(obj).encode()
                    path.write_bytes(raw)
                    modified['manual_action_plan_sha256'] = hashlib.sha256(raw).hexdigest()
                    modified['manual_action_plan'] = obj
                with self.assertRaisesRegex(ValueError, 'Unknown|profile mismatch'):
                    validate_varied_scope(modified)

    def test_scope_rejects_wrong_wire_speed_and_pose(self):
        for field in ('speed', 'target'):
            with self.subTest(field=field):
                scope = self.finalized_scope()
                action_plan = deepcopy(scope['manual_action_plan'])
                move = next(a for a in action_plan['actions'] if a['phase'] == 'approach_clear')
                if field == 'speed':
                    move['wire']['vel'] = 5.
                else:
                    move['target_pose'][2] += .1
                raw = json.dumps(action_plan).encode()
                (self.root / 'action-plan.json').write_bytes(raw)
                scope['manual_action_plan'] = action_plan
                scope['manual_action_plan_sha256'] = hashlib.sha256(raw).hexdigest()
                with self.assertRaisesRegex(ValueError, 'Continuous action'):
                    validate_varied_scope(scope)

    def test_actual_speed_monitors_remain_25_mm_and_5_degrees_per_second(self):
        session, robot, _ = self.session()
        session.phase = 'approach_clear'
        state = deepcopy(robot.s)
        state['tcp_speed'] = [25., 0.]
        state['joint_speeds_deg_s'][0] = 5.
        session.check_controller_state(state, moving=True)
        state['tcp_speed'][0] = 25.01
        with self.assertRaisesRegex(RuntimeError, 'Speed outside'):
            session.check_controller_state(state, moving=True)
        state['tcp_speed'][0] = 25.
        state['joint_speeds_deg_s'][0] = 5.01
        with self.assertRaisesRegex(RuntimeError, 'Speed outside'):
            session.check_controller_state(state, moving=True)

    def test_unknown_profile_fails_before_controller_or_capture(self):
        with patch('fr5_jev.prepare_varied_session.controller') as controller, \
                patch('fr5_jev.prepare_varied_session.capture') as capture:
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                build_manual_action_plan(self.route, [], joints(300), profile='bogus')
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                prepare(self.root / 'new', profile='bogus')
            controller.assert_not_called()
            capture.assert_not_called()

    def test_old_undeclared_legacy_artifacts_still_finalize_and_validate(self):
        scope = self.finalized_scope(self.legacy, old_legacy=True)
        self.assertEqual(scope['profile'], LEGACY_PROFILE)
        self.assertEqual(scope['move_speed'], 5.)
        self.assertEqual(scope['motion_speed_profile'], PROFILE_SPEEDS[LEGACY_PROFILE])
        self.assertNotIn('profile', validate_varied_scope(scope))

    def test_finalize_rejects_requested_or_embedded_profile_mismatch(self):
        geometry, plan = write_reviewed_geometry(self.root, self.route, self.continuous)
        with self.assertRaisesRegex(ValueError, 'differs'):
            finalize(self.root, profile=LEGACY_PROFILE)
        geometry['profile'] = LEGACY_PROFILE
        (self.root / 'geometry.json').write_text(json.dumps(geometry))
        with self.assertRaisesRegex(ValueError, 'profile mismatch'):
            finalize(self.root)
        self.assertFalse((self.root / 'scope.json').exists())

    def test_cli_passes_explicit_profile_and_finalization_can_inherit_it(self):
        result = {'status': 'synthetic', 'profile': CONTINUOUS_PROFILE,
                  'cup_change_between_observations_mm': 0., 'route_review_image': 'synthetic',
                  'action_plan': self.continuous}
        with patch('sys.argv', ['prepare', '--root', str(self.root), '--profile', CONTINUOUS_PROFILE]), \
                patch('fr5_jev.prepare_varied_session.prepare', return_value=result) as prepared, \
                redirect_stdout(io.StringIO()):
            main()
        prepared.assert_called_once_with(self.root.resolve(), profile=CONTINUOUS_PROFILE,
                                        prealigned=False,grasp_reference=None)
        with patch('sys.argv', ['prepare', '--root', str(self.root), '--finalize']), \
                patch('fr5_jev.prepare_varied_session.finalize', return_value={}) as finalized, \
                redirect_stdout(io.StringIO()):
            main()
        finalized.assert_called_once_with(self.root.resolve(), profile=None)

    def test_phase_review_uses_parent_selector_but_retains_visual_gates(self):
        session, _, _ = self.session()
        decision = {'workflow_review': {'cup_supported_and_released': True}}
        (self.root / 'review.json').write_text(json.dumps(decision))
        session.completed_stages = STAGES[:STAGES.index('retreat_clear')]
        with patch.object(session, 'phase_review_record', return_value=decision['workflow_review']) as select, \
                patch.object(TeacherSession, 'review_phase') as parent:
            session.review_phase('retreat_clear', 'review.json')
        select.assert_called_once_with(decision)
        parent.assert_called_once_with('retreat_clear', 'review.json')
        with patch.object(session, 'phase_review_record', return_value={}):
            with self.assertRaisesRegex(ValueError, 'supported released'):
                session.review_phase('retreat_clear', 'review.json')
        session.completed_stages = STAGES[:STAGES.index('lift_full')]
        with patch.object(session, 'phase_review_record', return_value={}):
            with self.assertRaisesRegex(ValueError, 'cup-follow'):
                session.review_phase('lift_full', 'review.json')


if __name__ == '__main__':
    unittest.main()
