"""Supervised whole-instruction pi0.5 rollout using direct model endpoints.

No teacher trajectories or phase prompts. Only --execute may dispatch, after a
fresh reviewed scope. No task-phase projection or small-step interpolation is
used by run(). Arm and gripper calls are serialized by the FR5 controller API.
"""
import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np

from .action_control import FR5ActionController, MotionLimits, finite_vector, rpc_vector, angle_error
from .feedback import gripper_feedback_valid
from .pi05_bridge import SSHPolicy, ADAPTER_SHA, observation_request, decoded_action, feedback_unchanged


def request_age(request):
    clock = request['source_clock']
    if clock not in ('host_monotonic', 'host_unix'):
        raise ValueError('Unknown observation clock')
    age = (time.monotonic() if clock == 'host_monotonic' else time.time())-request['source_time']
    if not np.isfinite(age) or age < 0:
        raise ValueError('Invalid observation age')
    return age


def validated_geometry(scope):
    vertices = np.asarray(scope['tool_vertices_flange_mm'], float)
    normal = np.asarray(scope['table_normal'], float)
    offset = float(scope['table_offset_mm'])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 8
            or not np.isfinite(vertices).all() or normal.shape != (3,)
            or not np.isfinite(normal).all() or abs(np.linalg.norm(normal)-1) > .01
            or not np.isfinite(offset)):
        raise ValueError('Invalid reviewed tool/table geometry')
    return vertices, normal, offset


def planned_action(raw, current, rpc, limits, joint_limits):
    raw = finite_vector(raw, 7, 'policy target')
    q = finite_vector(current['joints'], 6, 'measured joints')
    bounds = np.asarray(joint_limits, float)
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all():
        raise ValueError('Invalid joint limits')
    if not 0 <= raw[6] <= 100:
        raise ValueError('Predicted gripper outside0..100; raw output retained')
    if np.any(raw[:6] < bounds[:, 0]+3) or np.any(raw[:6] > bounds[:, 1]-3):
        raise ValueError('Predicted joint target violates joint limit margin')
    if not gripper_feedback_valid(current) or current.get('gripper_motiondone') == 3:
        raise ValueError('Invalid measured gripper feedback')
    pose = rpc_vector(rpc.call('GetForwardKin', raw[:6].tolist()), 6, 'FK')
    if np.any(pose[:3] < limits.workspace_min) or np.any(pose[:3] > limits.workspace_max):
        raise ValueError('Predicted endpoint outside reviewed workspace')
    origin = finite_vector(current['flange'], 6, 'measured flange')
    fraction = 1.
    for _ in range(24):
        candidate = q+fraction*(raw[:6]-q)
        small_pose = rpc_vector(rpc.call('GetForwardKin', candidate.tolist()), 6, 'FK')
        if (np.max(abs(candidate-q)) <= limits.joint_step_deg
                and np.linalg.norm(small_pose[:3]-origin[:3]) <= limits.translation_step_mm
                and np.max(abs(angle_error(small_pose[3:], origin[3:]))) <= limits.rotation_step_deg):
            if np.any(small_pose[:3] < limits.workspace_min) or np.any(small_pose[:3] > limits.workspace_max):
                raise ValueError('Interpolated endpoint outside reviewed workspace')
            # Match the linear-controller IK branch at its actual endpoint.
            solved = rpc_vector(rpc.call('GetInverseKinRef', 0, small_pose.tolist(), q.tolist()), 6, 'IK')
            if np.max(abs(solved-candidate)) > .05:
                raise ValueError('Interpolated target changes IK branch')
            break
        fraction *= .5
    else:
        raise ValueError('No bounded step toward predicted target')
    grip = float(current['gripper_position'])
    goal = float(grip+np.clip(raw[6]-grip, -limits.gripper_step_percent, limits.gripper_step_percent))
    return {'raw_target': raw.tolist(), 'fraction': fraction,
            'arm_target': [*candidate.tolist(), grip], 'gripper_target_percent': goal,
            'target_pose': small_pose.tolist(), 'raw_target_pose': pose.tolist()}


class PolicyController(FR5ActionController):
    def _gripper_complete(self,state,target,issued_monotonic):
        return (gripper_feedback_valid(state) and state['gripper_motiondone'] in (1,2)
                and time.monotonic()-issued_monotonic >= .8
                and (abs(state['gripper_position']-target) <= 1 or state['gripper_motiondone'] == 2))

    def ready_state(self, stationary=True):
        # User requested direct normal-speed execution without the additional
        # commissioning workspace/step restrictions. Retain actual device
        # faults, ownership, coordinate contract and live observation checks.
        state = self.stream.latest(max_age_s=.25)
        for key in ('joints','tcp','flange'):
            finite_vector(state[key],6,'measured '+key)
        for key in ('main_code','sub_code','EmergencyStop','collisionState',
                    'safety_stop0_state','safety_stop1_state','gripper_fault','gripper_fault_id'):
            if state[key] != 0:
                raise RuntimeError('Controller reports '+key)
        if state['rbtEnableState'] != 1 or state['tool'] != 0 or state['user'] != 0:
            raise RuntimeError('Controller not enabled in expected tool0/user0')
        if stationary and (state['program_state'] != 1 or state['robot_state'] != 1
                           or state['motion_done'] != 1 or state['mc_queue_len'] != 0):
            raise RuntimeError('Controller has not completed previous command')
        if not self.camera_fresh():
            raise RuntimeError('Live camera unavailable')
        if state['gripper_motiondone'] == 3:
            raise RuntimeError('Gripper reports lost or unstable contact')
        return state

    def _gate(self, generation=None):
        if not self.allow_motion or not self.armed or self.error:
            raise RuntimeError('Policy controller is stopped')
        if generation is not None and generation != self._generation:
            raise RuntimeError('Command cancelled')
        if time.monotonic() >= getattr(self, 'deadline', float('inf')):
            raise TimeoutError('Rollout time limit reached before dispatch')

    def _validate_arm_target(self, target, state):
        q=finite_vector(target[:6],6,'model joint target')
        if np.any(q < self.joint_limits[:,0]) or np.any(q > self.joint_limits[:,1]):
            raise ValueError('Model target outside actual controller joint limits')
        return rpc_vector(self.rpc.call('GetForwardKin',q.tolist()),6,'FK')

    def watchdog(self):
        if not self.armed:
            return
        try:
            self.ready_state(stationary=False)
        except Exception as exc:
            self.stop(str(exc))

    def _action_changes(self,target,state):
        kind=getattr(self,'_component',None)
        if kind == 'gripper':
            if np.max(abs(target[:6]-state['joints'])) > .05:
                raise RuntimeError('Arm changed before gripper command')
            target[:6]=state['joints']
            return False, target[6] != state['gripper_position']
        if kind == 'arm':
            if abs(target[6]-state['gripper_position']) > 1:
                raise RuntimeError('Gripper changed before arm command')
            target[6]=state['gripper_position']
            return bool(np.max(abs(target[:6]-state['joints'])) > 1e-7), False
        return (bool(np.max(abs(target[:6]-state['joints'])) > 1e-7),
                target[6] != state['gripper_position'])

    def execute(self,action,*,kind=None):
        target=finite_vector(action,7,'direct model action').copy()
        if not 0 <= target[6] <= 100:
            raise ValueError('Gripper target outside physical range')
        raw_gripper=float(target[6])
        target[6]=int(round(raw_gripper))
        self._component=kind
        try:
            result=super().execute(target)
            result['gripper_device_quantization']={'requested_percent':raw_gripper,
                                                   'wire_integer_percent':int(target[6])}
            return result
        finally:
            self._component=None

    def _append(self, target, kind, *, executed):
        row = super()._append(target, kind, executed=executed)
        with self._lock:
            self._journal[-1]['source'] = 'pi05_bounded_rollout'
        row['source'] = 'pi05_bounded_rollout'
        return row


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temporary.replace(path)


def validate_scope(scope):
    if scope.get('robot_ip') != '192.168.58.2' or scope.get('adapter_sha256') != ADAPTER_SHA:
        raise ValueError('Wrong robot or checkpoint')
    if not 0 <= time.time()-scope['reviewed_unix'] <= 600:
        raise ValueError('Scene review expired')
    if not scope.get('authorization') or scope.get('visible_local_workspace_reviewed') is not True:
        raise ValueError('Missing current review')
    if not 1 <= scope['max_steps'] <= 300 or not 5 <= scope['max_seconds'] <= 600:
        raise ValueError('Unbounded rollout')
    image = Path(scope['image'])
    if hashlib.sha256(image.read_bytes()).hexdigest() != scope['image_sha256']:
        raise ValueError('Reviewed image changed')
    return DirectSettings(speed_percent=scope['controller_speed_percent'])


@dataclass
class DirectSettings:
    speed_percent: float = 10.
    gripper_step_percent: float = 100.
    gripper_speed_percent: int = 20
    gripper_force_percent: int = 10
    command_timeout_s: float = 30.

    def __post_init__(self):
        if not np.isfinite(self.speed_percent) or not 0 < self.speed_percent <= 100:
            raise ValueError('Controller speed must be in0..100 percent')
        self.speed_percent = float(self.speed_percent)  # FR5 XML-RPC requires DOUBLE, including integer CLI defaults.


def decoded_whole_task_action(reply, request):
    """Return degrees and the original, unprojected gripper fraction.

    The shared bridge converts its gripper input to percent.  Give it a copy
    with the selected fraction projected first, so even very large *finite*
    model values cannot overflow during that conversion.  Its schema and
    checkpoint checks still apply, and the original reply remains untouched.
    """
    actions = np.asarray(reply.get('actions'), dtype=float)
    if actions.shape != (8, 7) or not np.isfinite(actions).all():
        raise ValueError('Expected eight finite seven-value actions')
    raw_fraction = float(actions[0, 6])
    projected_actions = actions.copy()
    projected_actions[0, 6] = np.clip(raw_fraction, 0., 1.)
    converted = decoded_action({**reply, 'actions': projected_actions}, request)
    return [*converted[:6], raw_fraction]


def direct_action(raw,current,rpc,joint_limits):
    """Adapt a model target in joint degrees and raw gripper fraction."""
    raw=finite_vector(raw,7,'model target')
    raw_fraction=float(raw[6])
    projected_fraction=float(np.clip(raw_fraction,0.,1.))
    projected_percent=projected_fraction*100.
    wire_integer_percent=int(round(projected_percent))
    if not gripper_feedback_valid(current) or current.get('gripper_motiondone') == 3:
        raise ValueError('Gripper feedback unavailable or contact lost')
    bounds=np.asarray(joint_limits,float)
    if bounds.shape != (6,2) or not np.isfinite(bounds).all():
        raise ValueError('Invalid controller joint limits')
    if np.any(raw[:6] < bounds[:,0]) or np.any(raw[:6] > bounds[:,1]):
        raise ValueError('Model target outside actual controller joint limits')
    pose=rpc_vector(rpc.call('GetForwardKin',raw[:6].tolist()),6,'FK')
    return {'raw_target':raw.tolist(),
            'raw_target_units':'joint_degrees_and_gripper_fraction',
            'gripper_adaptation':{'model_fraction':raw_fraction,
                                  'projected_fraction':projected_fraction,
                                  'projected_percent':projected_percent,
                                  'planned_wire_integer_percent':wire_integer_percent},
            'fraction':1.,
            'arm_target':[*raw[:6].tolist(),float(current['gripper_position'])],
            'gripper_target_percent':projected_percent,'target_pose':pose.tolist(),
            'mode':'direct_model_endpoint_no_phase_projection_no_step_interpolation'}


def run(root, execute=False):
    from .robot_fairino import FairinoFR5
    from .configuration_fairino import FairinoFR5Config
    from lerobot_camera_gemini import GeminiBridgeConfig
    from fr5_jev.stream_storage import StreamStorage
    root = Path(root).resolve()
    scope = json.loads((root/'scope.json').read_text())
    limits = validate_scope(scope)
    if not execute:
        return {'status': 'scope_validated_no_motion'}
    (root/'RUN_STARTED').open('x').close()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    policy = SSHPolicy(gpu=9)
    observer = controller = storage = None
    finished = threading.Event()
    report = {'task': scope['task'], 'adapter_sha256': ADAPTER_SHA, 'status': 'loading',
              'task_success_verified': False, 'human_success_required': True,
              'phase_prompt_switches': 0, 'teacher_trajectory_actions': 0, 'steps': [],
              'hold_rule': '5-second timed dwell on contact-feedback plus measured rise; not proof cup lifted'}
    def emit(x): print(json.dumps(x, ensure_ascii=False), flush=True)
    try:
        deadline = time.monotonic()+600
        while not policy.ready:
            if stop.is_set() or time.monotonic() > deadline:
                raise RuntimeError('Loading cancelled or timed out')
            try: emit(policy.wait_ready())
            except TimeoutError: emit({'status': 'loading'})
        validate_scope(scope)
        observer = FairinoFR5(FairinoFR5Config(camera=GeminiBridgeConfig(
            url='http://127.0.0.1:8765/snapshot_fast.npz', fps=6)))
        observer.connect()
        initial = observer.get_sample()['robot_state']
        if np.max(abs(np.asarray(initial['tcp'])-scope['initial_tcp'])) > .2:
            raise RuntimeError('Robot changed from reviewed start')
        if not gripper_feedback_valid(initial) or initial['gripper_motiondone'] != 1:
            raise RuntimeError('Initial gripper not ready and empty')
        controller = PolicyController(observer.stream,
            lambda: _camera_fresh(observer), limits, allow_motion=True)
        # Driver accepts one endpoint and one separate gripper command.
        # These are device dispatch settings, not learned or clipped targets.
        storage = StreamStorage.start_readonly(root, hashlib.sha256((root/'scope.json').read_bytes()).hexdigest(),
                                              required_bytes=128*1024**2)
        controller.arm(operator_confirmed=True)
        began = time.monotonic()
        controller.deadline = began+scope['max_seconds']
        # Monitor actual feedback and stop requests; no keyboard/deadman emulation.
        def watch():
            while not finished.wait(.04):
                if stop.is_set() or (root/'STOP').exists() or time.monotonic() >= controller.deadline:
                    controller.stop('Rollout stop requested')
                    return
                controller.watchdog()
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        contact_z = None
        dwell_done = False
        peak = 0.
        while len(report['steps']) < scope['max_steps'] and time.monotonic()-began < scope['max_seconds']:
            if stop.is_set() or (root/'STOP').exists() or controller.error:
                raise RuntimeError(controller.error or 'Stop requested')
            sample = observer.get_sample()
            request = observation_request(sample, scope['task'])
            reply = policy.predict(request)
            raw = decoded_whole_task_action(reply, request)
            current = observer.stream.latest(max_age_s=.25)
            feedback_unchanged(sample, current)
            if request_age(request) > 2:
                raise RuntimeError('Policy observation expired')
            row = {'index': len(report['steps']), 'request': request, 'reply': reply,
                   'observed_state': sample['robot_state'], 'commands': [], 'status': 'proposed'}
            path = root/f"step-{row['index']:04d}.json"
            save(path, row)
            storage.sync_files(path)
            # Persistence latency is never permission to use stale geometry.
            latest = observer.stream.latest(max_age_s=.25)
            feedback_unchanged(sample, latest)
            if request_age(request) > 3:
                raise RuntimeError('Proposal expired while saving evidence')
            plan = direct_action(raw, latest, controller.rpc, controller.joint_limits)
            row['plan'] = plan
            if contact_z is not None and latest['gripper_motiondone'] == 2:
                height = latest['tcp'][2]-contact_z
                peak = max(peak, height)
                if height >= 5 and not dwell_done:
                    started = time.monotonic()
                    emit({'status': 'five_second_dwell', 'cup_lift_verified': False})
                    while time.monotonic()-started < 5:
                        s = controller.ready_state()
                        if stop.is_set() or time.monotonic() >= controller.deadline or s['gripper_motiondone'] != 2:
                            raise RuntimeError('Dwell interrupted or contact lost')
                        if np.max(abs(np.asarray(s['joints'])-latest['joints'])) > .1:
                            raise RuntimeError('Unexpected movement during timed dwell')
                        time.sleep(.05)
                    report['dwell_seconds'] = time.monotonic()-started
                    dwell_done = True
                    # A fresh policy observation is required after the dwell.
                    row['status'] = 'dwell_completed_proposal_discarded'
                    save(path,row); storage.sync_files(path)
                    report['steps'].append({k:v for k,v in row.items() if k not in ('request','reply')})
                    continue
            save(path,row);storage.sync_files(path)
            if request_age(request) > 4:
                raise RuntimeError('Proposal expired before dispatch')
            before = controller.ready_state()
            feedback_unchanged(sample, before)
            # Prevalidate both components, then execute the same prediction in
            # two serial controller calls. No semantic phase substitutes values.
            controller._validate_arm_target(np.asarray(plan['arm_target']), before)
            result = controller.execute(plan['arm_target'],kind='arm')
            row['commands'].append(result)
            s = controller.ready_state()
            result = controller.execute([*s['joints'], plan['gripper_target_percent']],kind='gripper')
            row['commands'].append(result)
            after = observer.get_sample()
            # Keep the synchronized image/state pair intact, but do not use a
            # camera frame captured before device completion as current state.
            s = controller.ready_state()
            row['after_state'] = s
            row['after_camera_paired_state'] = after['robot_state']
            row['after_observation'] = observation_request(after, scope['task'])
            row['status'] = 'executed'
            if contact_z is None and s['gripper_motiondone'] == 2:
                contact_z = s['tcp'][2]
            save(path,row);storage.sync_files(path)
            report['steps'].append({k:v for k,v in row.items() if k not in ('request','reply','after_observation')})
            emit({'status':'policy_step_completed','index':row['index'],'tcp':s['tcp'],
                  'gripper':s['gripper_position'],'contact':s['gripper_motiondone'],
                  'rate_limit_fraction':plan['fraction']})
            if dwell_done and contact_z is not None and s['tcp'][2] <= contact_z+2 and s['gripper_motiondone'] == 1 and s['gripper_position'] >= 95:
                report['status'] = 'candidate_complete_pending_visual_and_user_review'
                break
        else:
            report['status'] = 'trial_limit_reached_task_not_verified'
        report.update(peak_tcp_rise_from_contact_mm=peak, five_second_dwell_completed=dwell_done)
    except BaseException as exc:
        report.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        if controller:
            controller.stop(str(exc))
        emit({'status':'rollout_stopped','error':report['error']})
    finally:
        finished.set()
        if controller:
            controller.stop('Rollout finished')
            report['actual_command_journal'] = list(controller._journal)
        save(root/'rollout-report.json',report)
        save(root/'summary.json',{'schema':'fr5-pi05-real-trial-summary-v1',
             'outcome':report['status'],'task':scope['task'],
             'task_success_verified':False,'training_ready':False,
             'completed_policy_steps':len(report['steps']),'report':'rollout-report.json'})
        if storage:
            try:
                storage.sync_tree()
                storage.frame_spool.close()
            except Exception as exc:
                report['finalization_error'] = str(exc)
                save(root/'rollout-report.json',report)
                storage.frame_spool.abort()
        if observer and observer.is_connected:
            observer.disconnect()
        policy.close()
    emit({'status':report['status'],'report':str(root/'rollout-report.json'),
          'task_success_verified':False,'steps':len(report['steps'])})
    return report


def _camera_fresh(observer):
    import urllib.request
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/status',timeout=.5) as response:
            c=json.load(response)['camera']
        return c['ok'] and 0 <= time.time()-c['received_at'] <= .5
    except Exception:
        return False


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    run(args.directory,args.execute)
