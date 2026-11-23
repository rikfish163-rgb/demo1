"""Offline tests for planned wire actions and the scene-review gate."""
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from .prepare_varied_session import build_manual_action_plan, finalize


class SyntheticIK:
    def GetInverseKinRef(self, mode, pose, reference):
        return [0, *pose]

    def GetForwardKin(self, joints):
        return [0, *joints]


class ManualActionPlanTests(unittest.TestCase):
    def test_actions_are_phase_ordered_and_exact_targets_have_matching_ik(self):
        def pose(z): return [0., 0., z, 0., 0., 0.]
        routes = {'approach_clear':[pose(200),pose(180)],
                  'close':[pose(180),pose(180)],
                  'lift_probe':[pose(180),pose(190)],
                  'return':[pose(190),pose(182),pose(180)],
                  'release':[pose(180),pose(180)]}
        rows=[]
        for phase, route in routes.items():
            for i, (a,b) in enumerate(zip(route,route[1:])):
                rows.extend([{'phase':phase,'segment':i,'fraction':f,'q':p}
                             for f,p in [(0.,a),(1.,b)]])
        with patch('fr5_jev.prepare_varied_session.controller',return_value=SyntheticIK()):
            plan=build_manual_action_plan(routes,rows,pose(200))
        actions=plan['actions']
        phases=[a['phase'] for a in actions]
        self.assertEqual(phases,sorted(phases,key=list(routes).index))
        self.assertEqual([a['wire']['pos'] for a in actions if a['phase']=='close'],[90,80,76])
        moves=[a for a in actions if a['kind']=='MoveL']
        for action in moves:
            self.assertEqual(action['wire']['joint_pos'],action['wire']['desc_pos'])
        approach=[a for a in moves if a['phase']=='approach_clear']
        self.assertEqual(approach[1]['source_pose'],approach[0]['target_pose'])
        returns=[a for a in moves if a['phase']=='return']
        self.assertEqual(returns[0]['wire']['vel'],3.)
        self.assertTrue(all(a['wire']['vel']==1. for a in returns[1:]))

    def test_finalize_never_invents_geometry_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=b'{}'
            (root/'action-plan.json').write_bytes(plan)
            geometry={'status':'prepared_for_visual_review','created_unix':time.time(),
                      'action_plan_sha256':hashlib.sha256(plan).hexdigest(),
                      'cup_tool_collision_samples':{'summary':{'finger_sample_overlap_count':18}}}
            (root/'geometry.json').write_text(json.dumps(geometry))
            with self.assertRaisesRegex(RuntimeError,'unresolved overlap'):
                finalize(root)
            geometry['cup_tool_collision_samples']['summary']['finger_sample_overlap_count']=0
            geometry['cup_tool_collision_samples']['approach_sweep_diagnostic']={
                'pose_count':10,'bodies':{name:{'poses_with_overlap':0} for name in ('-1','1','housing')}}
            (root/'geometry.json').write_text(json.dumps(geometry))
            with self.assertRaisesRegex(RuntimeError,'Missing independent'):
                finalize(root)
            self.assertFalse((root/'scope.json').exists())
            self.assertFalse((root/'approach_contact.json').exists())


if __name__=='__main__':
    unittest.main()
