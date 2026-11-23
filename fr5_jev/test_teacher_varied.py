"""Offline new-route safeguards, with synthetic states and mock RPC only."""
from copy import deepcopy
import hashlib,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
from .test_teacher_targets import FakeRobot,FakeRPC,FakeStorage
from .teacher_varied import VariedSession,STAGES,SPEEDS,validate_varied_scope

class VariedTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  start=[0,0,200,0,0,0];contact=[0,0,100,0,0,0]
  routes={k:[contact,contact] for k in STAGES}
  routes.update(approach_clear=[start,[0,0,180,0,0,0]],lift_probe=[contact,[0,0,110,0,0,0]],retreat_high=[[0,0,180,0,0,0],start])
  g={'minimum_joint_margin_deg':30,'max_fk_error_mm':0.,'max_fk_rotation_error_deg':0.,'expanded_tool_min_table_clearances_mm':{'tool':10.},'cup_center_change_mm':0.,'routes':routes}
  raw=json.dumps(g).encode();geo=self.root/'geometry.json';geo.write_bytes(raw)
  action_plan={'schema':'fr5-manual-action-plan-v2','actions':[{'id':'synthetic_'+phase,'phase':phase,'kind':'MoveGripper' if phase in ('close','release') else 'MoveL','wire':{}} for phase in STAGES]}
  action_plan['actions'][0].update(source_pose=start,wire=dict(joint_pos=[0,0,.9,0,90,0],desc_pos=[0,0,190,0,0,0],tool=0,user=0,vel=5.,acc=0.,ovl=100.,blendR=-1.,blendMode=0,exaxis_pos=[0.]*4,search=0,offset_flag=0,offset_pos=[0.]*6))
  action_raw=json.dumps(action_plan).encode();action_path=self.root/'action-plan.json';action_path.write_bytes(action_raw)
  self.scope=dict(robot_ip='192.168.58.2',contract='varied-cup-fixed-attitude-v1',phase_order=STAGES,max_lift_mm=50,closing_targets=[90,80,76],closing_speed=5,force=10,move_speed=1,max_session_s=1800,only_prealigned_instance=False,return_to_original_support=True,split_role='training_complete_trial',reviewed_at_monotonic=time.monotonic(),initial_tcp=start,contact_tcp=contact,authorization_source='SYNTHETIC TEST ONLY',evidence=['SYNTHETIC'],cup_unchanged_visually_verified=True,exterior_contact_route_reviewed=True,visible_local_sweep_clear=True,unseen_space_user_default_normal=True,no_visible_hands_anywhere=False,geometry_path=str(geo),geometry_sha256=hashlib.sha256(raw).hexdigest())
  self.scope.update(manual_action_plan_path=str(action_path),manual_action_plan_sha256=hashlib.sha256(action_raw).hexdigest(),manual_action_plan=action_plan)
  self.scope.update(move_speed=5,motion_speed_profile=SPEEDS,near_path_tracking_tolerance_mm=.3)
  (self.root/'scope.json').write_text(json.dumps(self.scope))
  self.robot=FakeRobot();self.robot.s.update(tcp=start.copy(),flange=start.copy(),gripper_position=100,gripper_motiondone=1,joints=[0.,0.,1.,0.,90.,0.])
  self.rpc=FakeRPC(self.robot);self.s=VariedSession(self.root,robot=self.robot,rpc=self.rpc)
  self.s.storage=FakeStorage()
  self.s.capture=SimpleNamespace(assert_healthy=Mock(),phase=Mock());self.s.limits=np.array([[-180,180]]*6)
  self.s.append([*self.robot.s['joints'],100],'hold',False)
 def tearDown(self):self.temp.cleanup()
 def test_empty_move_records_exact_sent_target(self):
  self.s.phase='approach_clear';self.s.move_to(self.s.planned_actions[0])
  moves=[x for x in self.rpc.calls if x[0]=='MoveL'];self.assertEqual(len(moves),1)
  self.assertEqual(self.s.journal[-1]['target'][:6],moves[0][1][0]);self.assertEqual(self.s.journal[-1]['target'][6],100)
  self.assertEqual(moves[0][1][4],5.)
  log=json.loads((self.root/'command-0002.json').read_text())
  self.assertEqual(log['planned_action']['id'],'synthetic_approach_clear')
  self.assertEqual(log['planned_action']['wire']['desc_pos'],moves[0][1][1])
  self.assertEqual(json.loads((self.root/'action-outcomes.json').read_text())[0]['status'],'completed')
 def test_timeout_covers_actual_distance_at_frozen_speed(self):
  self.s.phase='approach_clear'
  self.assertEqual(self.s.command_timeouts('arm',self.robot.s,[0,0,185,0,0,0]),(11.,14.))
  self.s.phase='approach_contact'
  self.assertEqual(self.s.command_timeouts('arm',self.robot.s,[0,0,195,0,0,0]),(15.,18.))
 def test_empty_route_rejects_contact_and_excessive_segment_before_send(self):
  self.s.phase='approach_clear'
  action=deepcopy(self.s.planned_actions[0]);action['wire']['desc_pos'][2]=180
  with self.assertRaises(ValueError):self.s.move_to(action)
  self.robot.s['gripper_motiondone']=2
  with self.assertRaises(RuntimeError):self.s.move_to(self.s.planned_actions[0])
  self.assertFalse(self.rpc.calls)
 def test_unplanned_wire_arguments_are_refused_before_dispatch(self):
  self.s.phase='approach_clear';a=self.s.planned_actions[0]
  wire=a['wire'];args=(wire['joint_pos'],wire['desc_pos'],0,0,20.,0.,100.,-1.,0,[0.]*4,0,0,[0.]*6)
  with self.assertRaisesRegex(RuntimeError,'differ'):
   self.s.send('MoveL',args,[*wire['joint_pos'],100],kind='arm',destination_pose=wire['desc_pos'])
  self.assertFalse(self.rpc.calls)
 def test_changed_start_and_changed_ik_are_refused_before_dispatch(self):
  self.s.phase='approach_clear';a=deepcopy(self.s.planned_actions[0]);a['source_pose'][2]-=1
  with self.assertRaisesRegex(RuntimeError,'start'):self.s.move_to(a)
  a=deepcopy(self.s.planned_actions[0]);a['wire']['joint_pos'][2]+=1
  with self.assertRaisesRegex(RuntimeError,'branch'):self.s.move_to(a)
  self.assertFalse(any(call[0]=='MoveL' for call in self.rpc.calls))
 def test_wrong_corridor_and_overshoot_stop(self):
  self.s.phase='approach_clear';s=deepcopy(self.robot.s);s['tcp'][0]=.6
  with self.assertRaises(RuntimeError):self.s.check_controller_state(s,moving=True)
  s=deepcopy(self.robot.s);s['tcp'][2]=189.4
  with self.assertRaises(RuntimeError):self.s.check_segment_progress(s,self.robot.s,[0,0,190,0,0,0])
 def test_grasp_and_release_pose_are_required(self):
  self.s.phase='lift_probe'
  with self.assertRaises(RuntimeError):self.s.check_command_permission('arm',self.robot.s,None)
  self.s.phase='release'
  with self.assertRaises(RuntimeError):self.s.check_command_permission('gripper',self.robot.s,None)
  self.s.phase='retreat_high'
  with self.assertRaises(RuntimeError):self.s.check_command_permission('arm',self.robot.s,None)
 def test_near_contact_corridor_remains_tightly_bounded(self):
  self.s.phase='lift_probe';s=deepcopy(self.robot.s);s['tcp']=[.31,0,105,0,0,0]
  with self.assertRaises(RuntimeError):self.s.check_controller_state(s,moving=True)
 def test_external_scene_hold_blocks_new_commands(self):
  (self.root/'MOTION_HOLD').write_text('SYNTHETIC scene changed')
  with self.assertRaises(RuntimeError):validate_varied_scope(self.scope)
  self.s.phase='approach_clear'
  with self.assertRaises(RuntimeError):self.s.check_command_permission('arm',self.robot.s,None)
 def test_phase_skip_replay_and_success_shortcut_rejected(self):
  with self.assertRaises(ValueError):self.s.review_phase('close','missing.json')
  with self.assertRaises(RuntimeError):self.s.finish('success',['synthetic'])
  with self.assertRaises(RuntimeError):self.s.finish('pending_review',['synthetic'])
 def test_tamper_age_and_orientation_refused(self):
  s=deepcopy(self.scope);s['reviewed_at_monotonic']-=301
  with self.assertRaises(ValueError):validate_varied_scope(s)
  (self.root/'geometry.json').write_text('{}')
  with self.assertRaises(ValueError):validate_varied_scope(self.scope)

if __name__=='__main__':unittest.main()
