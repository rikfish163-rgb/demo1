"""Fresh-scene, fixed-attitude full cup demonstrations; inert on import.

Only the scope's ordered route is available. No arbitrary pose, replay, reset,
enable, force increase, or joint-trajectory replay. Empty and held routes have
separate checks; all training actions are the MoveL/MoveGripper targets sent.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import sys
import time
import numpy as np
from fr5_runtime.robot import HOST, require_ready, use_shared_state_stream
from fr5_runtime.session_guard import require_session_open
from lerobot_robot_fairino.feedback import gripper_feedback_valid
from .teacher_targets import TeacherSession, angle_delta, review_age, save
from .observe import capture as observe

STAGES = ['approach_clear','approach_near','approach_contact','close','lift_probe',
          'lift_full','return','release','retreat_probe','retreat_clear','retreat_high']
HELD = {'lift_probe','lift_full','return'}
EMPTY = {'initial_hold','approach_clear','approach_near','approach_contact',
         'retreat_probe','retreat_clear','retreat_high'}
from .motion_profiles import (SPEEDS, LEGACY_PROFILE, CONTINUOUS_PROFILE, PROFILE_SPEEDS,
                              FREE_PHASES, PROBE_PHASES, validate_profile, record_profile,
                              matching_profiles, motion_limits)


def validate_continuous_actions(actions, support_z, speeds):
    for action in actions:
        if action['kind'] != 'MoveL':
            continue
        wire = action['wire']
        try:
            source = np.asarray(action['source_pose'], float)
            target = np.asarray(wire['desc_pos'], float)
            if (source.shape != (6,) or target.shape != (6,)
                    or not np.isfinite(source).all() or not np.isfinite(target).all()
                    or action['target_pose'] != wire['desc_pos']
                    or action['joint_target_deg'] != wire['joint_pos']):
                raise ValueError('Continuous action target or source mismatch')
            maximum, category = motion_limits(CONTINUOUS_PROFILE, action['phase'], source,
                                              target, support_z)
            distance = float(np.linalg.norm(target[:3] - source[:3]))
            speed = speeds[category]
            if (not .03 < distance <= maximum + 1e-8 or wire['vel'] != speed
                    or max(8., 2. * distance / speed + 5.) > 35.1):
                raise ValueError('Continuous action exceeds its phase or time limit')
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError('Incomplete continuous action') from exc


def distance_to_segment(p, a, b):
    d = np.asarray(b)-a
    t = float(np.dot(np.asarray(p)-a,d)/np.dot(d,d)) if np.dot(d,d)>1e-12 else 0.
    return float(np.linalg.norm(np.asarray(p)-(a+np.clip(t,0,1)*d))),t


def validate_varied_scope(s):
    profile = record_profile(s)
    speeds = PROFILE_SPEEDS[profile]
    prealigned = s.get('prealigned_start', False)
    if (s.get('robot_ip') != HOST or s.get('contract') != 'varied-cup-fixed-attitude-v1'
            or s.get('phase_order') != STAGES or s.get('max_lift_mm') != 50
            or s.get('closing_targets') != [90,80,76] or s.get('closing_speed') != 5
            or s.get('force') != 10 or s.get('move_speed') != speeds['empty_free']
            or s.get('motion_speed_profile') != speeds
            or s.get('near_path_tracking_tolerance_mm') != .3
            or s.get('max_session_s') != 1800 or s.get('only_prealigned_instance') is not prealigned
            or s.get('return_to_original_support') is not True
            or s.get('split_role') not in ('training_complete_trial','held_out_complete_trial')):
        raise ValueError('Invalid full-demonstration scope')
    if not 0 <= review_age(s) <= 300:
        raise ValueError('Stale scene review')
    for name in ('initial_tcp','contact_tcp'):
        p = np.array(s.get(name),float)
        if p.shape != (6,) or not np.isfinite(p).all():
            raise ValueError('Invalid finite pose')
    start_offset = np.linalg.norm(np.array(s['initial_tcp'][:3])-s['contact_tcp'][:3])
    if prealigned:
        if (profile != CONTINUOUS_PROFILE or start_offset > .001
                or s.get('prealigned_visual_entry_reviewed') is not True
                or s.get('requires_runtime_visual_alignment') is not True):
            raise ValueError('Prealigned start needs the reviewed actual contact pose')
    elif not 60 <= start_offset <= 260:
        raise ValueError('Initial offset outside reviewed60..260mm region')
    if not s.get('authorization_source') or not s.get('evidence'):
        raise ValueError('Missing authorization/evidence')
    supervised = s.get('execution_basis') == 'operator_supervised_nominal'
    if supervised and (profile != CONTINUOUS_PROFILE or s.get('requires_runtime_visual_alignment') is not True
            or s.get('exterior_contact_route_reviewed') is not False
            or s.get('visible_local_sweep_clear') is not False
            or not s.get('current_scene_review',{}).get('onsite_clearance_source')):
        raise ValueError('Supervised geometry assumptions must remain explicit')
    geometry_flags = ('nominal_contact_route_reviewed',) if supervised else ('exterior_contact_route_reviewed','visible_local_sweep_clear')
    for flag in ('cup_unchanged_visually_verified','unseen_space_user_default_normal',*geometry_flags):
        if s.get(flag) is not True:
            raise ValueError('Missing current evidence: '+flag)
    if s.get('no_visible_hands_anywhere') is not False:
        raise ValueError('Do not mislabel the person at the far left as absent')
    require_session_open(Path(s['geometry_path']).parent)
    raw = Path(s['geometry_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != s.get('geometry_sha256'):
        raise ValueError('Geometry provenance mismatch')
    action_plan_path = Path(s.get('manual_action_plan_path',''))
    if not action_plan_path.is_file():
        raise ValueError('Missing precomputed manual action plan')
    action_plan_raw = action_plan_path.read_bytes()
    if hashlib.sha256(action_plan_raw).hexdigest() != s.get('manual_action_plan_sha256'):
        raise ValueError('Manual action plan provenance mismatch')
    action_plan = json.loads(action_plan_raw)
    matching_profiles(s, action_plan)
    if (action_plan.get('schema') != 'fr5-manual-action-plan-v2'
            or action_plan != s.get('manual_action_plan')
            or not isinstance(action_plan.get('actions'), list)
            or any(action.get('phase') not in STAGES
                   or action.get('kind') not in ('MoveL', 'MoveGripper')
                   for action in action_plan['actions'])):
        raise ValueError('Manual action plan contract mismatch')
    actions = action_plan['actions']
    ids = [a.get('id') for a in actions]
    if (not actions or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)
            or [STAGES.index(a['phase']) for a in actions] != sorted(STAGES.index(a['phase']) for a in actions)
            or any(not isinstance(a.get('wire'), dict) for a in actions)):
        raise ValueError('Manual actions must be unique, ordered and contain wire arguments')
    g = json.loads(raw)
    if prealigned:
        if (any(a['phase'] in STAGES[:3] for a in actions)
                or any(np.max(abs(np.asarray(p)-s['initial_tcp'])) > .001
                       for phase in STAGES[:3] for p in g['routes'][phase])):
            raise ValueError('Prealigned entry cannot contain an unreviewed approach move')
    matching_profiles(s, action_plan, g)
    if isinstance(g.get('action_plan'), dict):
        matching_profiles(s, action_plan, g, g['action_plan'])
    if profile == CONTINUOUS_PROFILE:
        validate_continuous_actions(actions, s['contact_tcp'][2], speeds)
    kin = g.get('kinematics', g)
    cup_center_change = g.get('cup_center_change_mm', g.get('cup_change_between_observations_mm'))
    if (kin['minimum_joint_margin_deg'] < 3 or kin['max_fk_error_mm'] > .2
            or kin['max_fk_rotation_error_deg'] > .05
            or min(g['expanded_tool_min_table_clearances_mm'].values()) < 5
            or cup_center_change is None or cup_center_change > 2
            or set(g['routes']) != set(STAGES)):
        raise ValueError('Route geometry failed')
    if np.max(abs(np.array(g['routes']['approach_clear'][0])-s['initial_tcp']))>.001:
        raise ValueError('Geometry start mismatch')
    for phase, route in g['routes'].items():
        for index, p in enumerate(route):
            # A fresh varied start may differ in roll/pitch from the level
            # contact attitude.  Only the first measured start point is
            # exempt; every planned point after it must use the reviewed level
            # attitude.  The sampled MoveL path below still checks continuity.
            if ((phase == 'approach_clear' and index == 0)
                    or (phase == 'retreat_high' and index == len(route)-1)):
                expected = (s.get('terminal_tcp') if prealigned and phase == 'retreat_high'
                            else s['initial_tcp'])
                if expected is None or np.max(abs(np.array(p)-expected)) > .001:
                    raise ValueError('Route endpoint differs from the measured start pose')
                continue
            if len(p)!=6 or not np.isfinite(p).all() or max(abs(angle_delta(p[3:],s['contact_tcp'][3:])))>.15:
                raise ValueError('Scope contains an unreviewed orientation')
    return g


class VariedSession(TeacherSession):
    validate_scope = staticmethod(validate_varied_scope)

    def __init__(self, directory, **kwargs):
        super().__init__(directory, **kwargs)
        if (self.scope.get('requires_runtime_visual_alignment') is True
                and not getattr(self, 'supports_visual_checkpoints', False)):
            raise RuntimeError('This supervised scope must use the continuous visual workflow')
        self.geometry = validate_varied_scope(self.scope)
        self.origin = np.asarray(self.scope['contact_tcp'],float)
        self.completed_stages = []
        self.segment_checks = None
        self.planned_actions = self.scope['manual_action_plan']['actions']
        self.phase_action_index = 0
        self.action_outcomes = []
        self.task_description = 'Grasp the cup, lift it50mm, put it back, release and withdraw'

    def arm_requires_hold(self):
        return self.phase in HELD

    def record_prealigned_entry(self, phase):
        """Record an already observed start; never pretend an approach was sent."""
        if (self.scope.get('prealigned_start') is not True or phase not in STAGES[:3]
                or phase != STAGES[len(self.completed_stages)]
                or any(a['phase'] == phase for a in self.planned_actions)):
            raise RuntimeError('Invalid prealigned entry or phase replay')
        state=self.state()
        if (np.max(abs(np.asarray(state['tcp'])-self.origin)) > .1
                or state['gripper_position'] != 100 or state['gripper_motiondone'] != 1):
            raise RuntimeError('Visually reviewed prealigned start changed')
        self.phase=phase
        self.capture.phase(phase, teacher={'policy':'observed_prealigned_start',
                                          'motion_commands_sent':0})
        self.completed_stages.append(phase)
        save(self.root/'completed-stages.json',self.completed_stages)
        save(self.root/f'prealigned-{phase}.json',{'status':'observed_start_no_command',
                                                'state':state,'scope_sha256':self.scope_sha})
        return state

    def move_speed(self, destination_pose, source_pose=None):
        if source_pose is None:
            actions = [a for a in self.planned_actions if a['phase'] == self.phase]
            source_pose = (actions[self.phase_action_index]['source_pose']
                           if self.phase_action_index < len(actions)
                           and 'source_pose' in actions[self.phase_action_index]
                           else destination_pose)
        _, category = motion_limits(record_profile(self.scope), self.phase, source_pose,
                                    destination_pose, self.origin[2])
        return self.scope['motion_speed_profile'][category]

    def command_timeouts(self, kind, initial, destination_pose):
        if kind != 'arm':return super().command_timeouts(kind,initial,destination_pose)
        distance=float(np.linalg.norm(np.array(destination_pose[:3])-initial['tcp'][:3]))
        # Previous observed wire vel1 moves1mm/s, and blocking MoveL replies
        # only at completion. Budget twice nominal travel plus5s, bounded35s.
        timeout=max(8.,2*distance/self.move_speed(destination_pose, initial['tcp'])+5.)
        if timeout>35.1:raise ValueError('Segment needs an unreviewed time budget')
        return timeout,timeout+3.

    def route(self):
        if self.phase == 'initial_hold':
            return [self.expected_start_pose,self.expected_start_pose]
        return self.geometry['routes'][self.phase]

    def check_controller_state(self,s,*,moving=False,held_position=None,gripper_reference=None):
        require_ready(s,stationary=not moving)
        if (s['robot_mode']!=1 or s['robot_state'] in (3,4) or s['gripper_active']!=1
                or s['gripper_fault'] or s['gripper_fault_id']):
            raise RuntimeError('Controller or gripper unavailable')
        pose=np.array(s['tcp']);route=np.array(self.route())
        tolerance=.5 if self.phase in ('approach_clear','approach_near','retreat_clear','retreat_high') else .3
        orientation_error=float(max(abs(angle_delta(pose[3:],self.origin[3:]))))
        orientation_limit=.15
        if self.phase in ('initial_hold','approach_clear','retreat_high'):
            orientation_limit=max(
                orientation_limit,
                float(max(abs(angle_delta(self.expected_start_pose[3:],self.origin[3:]))))+.2)
        if (min(distance_to_segment(pose[:3],a[:3],b[:3])[0] for a,b in zip(route,route[1:]))>tolerance
                or orientation_error>orientation_limit):
            raise RuntimeError('Left reviewed phase corridor')
        if s['tcp_speed'][0]>25 or max(abs(v) for v in s['joint_speeds_deg_s'])>5:
            raise RuntimeError('Speed outside monitored limit')
        if self.phase in EMPTY and (not gripper_feedback_valid(s) or s['gripper_position']!=100 or s['gripper_motiondone']!=1):
            raise RuntimeError('Empty route requires confirmed fully open gripper')
        if held_position is not None and (not gripper_feedback_valid(s) or s['gripper_motiondone']!=2
                or abs(s['gripper_position']-held_position)>1):
            raise RuntimeError('Stable holding feedback lost')
        if gripper_reference is not None and (moving or np.max(abs(pose[:3]-np.array(gripper_reference)[:3]))>.15):
            raise RuntimeError('Arm moved during gripper operation')
        return float(pose[2]-self.origin[2])

    def check_command_permission(self,kind,initial,destination_pose):
        require_session_open(Path(self.scope['geometry_path']).parent)
        if kind=='arm':
            if self.phase in HELD:
                if self.holding_position is None or self.released:
                    raise RuntimeError('Held route lacks stable grasp')
            elif self.phase not in EMPTY or self.phase=='initial_hold':
                raise RuntimeError('No arm action in this phase')
            if self.phase.startswith('retreat') and not self.released:
                raise RuntimeError('Retreat before supported release')
        elif kind=='gripper':
            if self.phase not in ('close','release') or np.max(abs(np.array(initial['tcp'][:3])-self.origin[:3]))>.05:
                raise RuntimeError('Gripper change requires reviewed exterior support pose')

    def check_segment_progress(self,s,initial,destination_pose):
        p=np.array(s['tcp'][:3]);a=np.array(initial['tcp'][:3]);b=np.array(destination_pose[:3])
        distance,t=distance_to_segment(p,a,b)
        tolerance=.5 if self.phase in ('approach_clear','approach_near','retreat_clear','retreat_high') else .3
        if distance>tolerance or not -tolerance <= t*np.linalg.norm(b-a) <= np.linalg.norm(b-a)+tolerance:
            raise RuntimeError('Segment tracking/overshoot failed')
        if self.segment_checks:
            progress=[x['fraction'] for x in self.segment_checks]
            q=np.array([x['q'] for x in self.segment_checks])
            expected=np.array([np.interp(t,progress,q[:,j]) for j in range(6)])
            if max(abs(np.array(s['joints'])-expected))>.6:
                raise RuntimeError('Joint branch departed from sampled segment')

    def review_phase(self,phase,decision_id):
        if len(self.completed_stages)>=len(STAGES) or phase!=STAGES[len(self.completed_stages)]:
            raise ValueError('Phase order or replay rejected')
        decision=json.loads((self.root/decision_id).read_text())
        review=self.phase_review_record(decision)
        if phase in ('lift_full','return') and review.get('cup_follow_visually_verified') is not True:
            raise ValueError('Held continuation requires visual cup-follow evidence')
        if phase.startswith('retreat') and review.get('cup_supported_and_released') is not True:
            raise ValueError('Retreat requires supported released cup evidence')
        super().review_phase(phase,decision_id)

    def send(self, method, args, target, *, kind, destination_pose=None):
        actions = [a for a in self.planned_actions if a['phase'] == self.phase]
        if self.phase_action_index >= len(actions):
            raise RuntimeError('No remaining planned action for this phase')
        action = actions[self.phase_action_index]
        wire = action['wire']
        if method == 'MoveL':
            keys = ('joint_pos','desc_pos','tool','user','vel','acc','ovl','blendR',
                    'blendMode','exaxis_pos','search','offset_flag','offset_pos')
        elif method == 'MoveGripper':
            keys = ('index','pos','vel','force','maxtime','block','type','rotNum','rotVel','rotTorque')
        else:
            raise RuntimeError('Unplanned controller method')
        if action['kind'] != method or list(args) != [wire[k] for k in keys]:
            raise RuntimeError('Actual controller arguments differ from the planned action')
        if ((method == 'MoveL' and (kind != 'arm' or list(target[:6]) != wire['joint_pos']
                                    or list(destination_pose) != wire['desc_pos']
                                    or wire['vel'] != self.move_speed(destination_pose, action['source_pose'])))
                or (method == 'MoveGripper' and (kind != 'gripper' or target[6] != wire['pos']))):
            raise RuntimeError('Planned action has inconsistent target, kind or speed')
        result = super().send(method, args, target, kind=kind, destination_pose=destination_pose,
                              action_metadata={'id': action['id'],
                                               'plan_sha256': self.scope['manual_action_plan_sha256'],
                                               'wire': wire})
        self.phase_action_index += 1
        self.action_outcomes.append({'id': action['id'], 'status': 'completed'})
        save(self.root/'action-outcomes.json', self.action_outcomes)
        return result

    def move_to(self,action):
        if action.get('phase') != self.phase or action.get('kind') != 'MoveL':
            raise ValueError('Move must be the planned action for this phase')
        s=self.state(held=self.arm_requires_hold());pose=np.array(action['wire']['desc_pos'],float)
        source=np.array(action['source_pose'],float)
        if (np.linalg.norm(np.asarray(s['tcp'][:3])-source[:3]) > .15
                or max(abs(angle_delta(s['tcp'][3:],source[3:]))) > .2):
            raise RuntimeError('Actual pose differs from planned action start')
        phase_actions = [a for a in self.planned_actions if a['phase'] == self.phase]
        if (self.phase_action_index >= len(phase_actions)
                or action['source_pose'] != phase_actions[self.phase_action_index].get('source_pose')):
            raise RuntimeError('Move action start differs from the reviewed plan')
        if 'target_pose' in action and action['target_pose'] != action['wire']['desc_pos']:
            raise RuntimeError('Move action target differs from the reviewed wire pose')
        distance=np.linalg.norm(pose[:3]-s['tcp'][:3])
        profile = record_profile(self.scope)
        if profile == LEGACY_PROFILE:
            maximum = 15.05
        else:
            maximum, category = motion_limits(profile, self.phase, source, pose, self.origin[2])
            actual_maximum, actual_category = motion_limits(profile, self.phase, s['tcp'], pose, self.origin[2])
            if category != actual_category:
                raise ValueError('Actual MoveL start crosses the return speed boundary')
            maximum = min(maximum, actual_maximum) + 1e-8
            if np.linalg.norm(pose[:3]-source[:3]) > maximum:
                raise ValueError('Planned MoveL target exceeds the phase limit')
            # Controller completion already accepts <=.05mm endpoint error.
            # Preserve the exact planned bound; allow that same measurement
            # tolerance when measuring distance from the live start sample.
            maximum += .05
        if not .03<distance<=maximum:
            raise ValueError('Actual MoveL target exceeds the phase distance limit')
        reference=s['joints'];checks=[{'fraction':0.,'q':reference,'pose':s['tcp']}]
        for alpha in np.linspace(0,1,max(2,math.ceil(distance)+1))[1:]:
            p=np.array(s['tcp'])+alpha*(pose-s['tcp'])
            ik=self.rpc.call('GetInverseKinRef',0,p.tolist(),reference)
            if len(ik)!=7 or ik[0]!=0:raise RuntimeError('Fresh segment IK failed')
            q=np.array(ik[1:]);fk=self.rpc.call('GetForwardKin',q.tolist())
            if (not np.isfinite(q).all() or np.any(q<self.limits[:,0]+3) or np.any(q>self.limits[:,1]-3)
                    or max(abs(q-reference))>1 or abs(math.sin(math.radians(q[4])))<.2):
                raise RuntimeError('Fresh joint branch/margin failed')
            if (len(fk)!=7 or fk[0]!=0 or np.linalg.norm(np.array(fk[1:4])-p[:3])>.2
                    or max(abs(angle_delta(fk[4:],p[3:])))>.05):
                raise RuntimeError('Fresh IK/FK mismatch')
            checks.append({'fraction':float(alpha),'q':q.tolist(),'pose':p.tolist()});reference=q.tolist()
        self.segment_checks=checks
        planned_q=np.array(action['wire']['joint_pos'],float)
        if planned_q.shape != (6,) or max(abs(planned_q-reference)) > .1:
            raise RuntimeError('Fresh IK branch differs from the planned wire joints')
        save(self.root/f'path-{self.command_sequence+1:04d}.json',{'checks':checks})
        wire=action['wire']
        result=self.send('MoveL',(wire['joint_pos'],wire['desc_pos'],wire['tool'],wire['user'],
                         wire['vel'],wire['acc'],wire['ovl'],wire['blendR'],wire['blendMode'],
                         wire['exaxis_pos'],wire['search'],wire['offset_flag'],wire['offset_pos']),
                         [*planned_q,self.target[6]],kind='arm',destination_pose=pose)
        self.segment_checks=None
        if self.phase in HELD:self.peak_height=max(self.peak_height,float(result['tcp'][2]-self.origin[2]))
        return result

    def run_phase(self):
        phase=self.phase
        self.phase_action_index=0
        actions=[a for a in self.planned_actions if a['phase']==phase]
        if not actions:
            raise RuntimeError('No planned actions for this phase')
        if phase=='close' and [a['wire'].get('pos') for a in actions] != self.scope['closing_targets']:
            raise RuntimeError('Closing plan differs from bounded closing targets')
        if phase=='release' and (len(actions) != 1 or actions[0]['wire'].get('pos') != 100):
            raise RuntimeError('Release plan differs from bounded release target')
        if phase=='close':result=self.close_gripper()
        elif phase=='release':result=self.release()
        else:
            result=self.state(held=self.arm_requires_hold())
            for action in actions:
                result=self.move_to(action)
            if phase=='return':
                if np.max(abs(np.array(result['tcp'][:3])-self.origin[:3]))>.05:
                    raise RuntimeError('Original support pose not reached')
                self.returned=True
        if phase=='close' and self.holding_position is not None:
            for action in actions[self.phase_action_index:]:
                self.action_outcomes.append({'id':action['id'], 'status':'skipped_after_contact'})
            self.phase_action_index=len(actions)
            save(self.root/'action-outcomes.json', self.action_outcomes)
        if self.phase_action_index != len(actions):
            raise RuntimeError('Phase ended without accounting for every planned action')
        self.completed_stages.append(phase)
        save(self.root/'completed-stages.json',self.completed_stages)
        return result

    def final_summary_fields(self):
        return dict(completed_stages=list(self.completed_stages), split_role=self.scope['split_role'],
                    start_offset_mm=float(np.linalg.norm(self.expected_start_pose[:3]-self.origin[:3])))

    def finish(self,outcome='interrupted',evidence=None):
        if outcome=='success':
            raise RuntimeError('The user must review the completed episode; finish as pending_review')
        if outcome=='pending_review' and (self.completed_stages!=STAGES or self.peak_height<49.9
                                           or not self.released or self.failed or not evidence):
            raise RuntimeError('Pending review needs a completed approach/lift/place/release/withdraw and evidence')
        return super().finish(outcome,evidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();session=VariedSession(args.directory)
    if not args.execute:print('Scope valid; no connection or motion');return
    signal.signal(signal.SIGINT,lambda *_:session.interrupted.set())
    signal.signal(signal.SIGTERM,lambda *_:session.interrupted.set())
    finished=False
    try:
        state=session.start()
        with use_shared_state_stream(session.robot.stream):
            print(json.dumps({'status':'recording_ready_no_motion','state':state}),flush=True)
            for line in sys.stdin:
                request=json.loads(line);op=request['op']
                if op=='finish':
                    result=session.finish(request.get('outcome','interrupted'),request.get('evidence'))
                    finished=True;print(json.dumps({'status':'finished','summary':result}),flush=True);break
                if op=='observe':
                    if not re.fullmatch('[a-z0-9-]+',request['name']):raise ValueError('Invalid observation name')
                    result=session.observe_bounded(session.root/request['name'],observer=observe)
                    print(json.dumps({'status':'observed','image':str(session.root/request['name']/'color.jpg'),'state':result['robot_after']}),flush=True);continue
                if op=='decide':
                    result=session.decide(request.get('phase',request.get('skill')),request['decision_id'])
                    print(json.dumps({'status':'jev_decided_no_action',**result}),flush=True);continue
                if op not in STAGES:raise ValueError('Only scoped stages allowed')
                (session.root/f'PHASE_{op}').open('x').close()
                session.review_phase(op,request['decision_id']);result=session.run_phase()
                print(json.dumps({'status':'phase_completed','phase':op,'state':result}),flush=True)
    except BaseException as exc:
        if session.storage is not None or session.active_command:
            session.stop_once(exc,arm_in_flight=session.active_command)
        print(json.dumps({'status':'failed_no_retry','error':f'{type(exc).__name__}: {exc}'}),flush=True)
        raise
    finally:
        try:
            if not finished and session.writer:session.finish('interrupted')
        finally:
            if session.robot.is_connected:session.robot.disconnect()


if __name__=='__main__':main()
