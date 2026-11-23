"""Pi0.5 observation/action bridge; motion requires an already-armed bounded executor.

The CLI previews proposals only. The step API can execute one bounded action
through the existing FR5ActionController, then obtains a new paired observation.
It never enables the robot, arms a controller, retries motion, or labels success.
"""
import argparse
import base64
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import select
import shlex
import subprocess
import time
import uuid

import numpy as np

from .feedback import gripper_feedback_valid
from lerobot_camera_gemini.timing import camera_clock, MONOTONIC

ADAPTER_SHA = '3d022d30b94634f580c1e882d9bf313a65c722327003b3332f1facec8ca29cbf'
SERVER = '/home/data4/hts/demo1/fr5-learning/runtime/serve_pi05_fr5_stdio.py'
PYTHON = '/home/data4/hts/demo1/fr5-learning/.venv/bin/python'
ARM_PHASES = {'approach_clear', 'approach_near', 'approach_contact', 'lift_probe',
              'lift_full', 'return', 'retreat_probe', 'retreat_clear', 'retreat_high'}


def solve_tcp_target(pose_mm_deg, current, rpc):
    """Read-only tool0/user0 IK using current joints as the solution reference."""
    pose = np.asarray(pose_mm_deg, float)
    joints = np.asarray(current['joints'], float)
    if (pose.shape != (6,) or joints.shape != (6,) or not np.isfinite(pose).all()
            or not np.isfinite(joints).all() or current['tool'] != 0 or current['user'] != 0):
        raise ValueError('Need finite XYZ/RxRyRz and current joints in tool0/user0')
    if np.max(abs(np.asarray(current['tcp'])-current['flange'])) > .1:
        raise ValueError('This IK helper requires tool0 to match the flange')
    result = rpc.call('GetInverseKinRef', 0, pose.tolist(), joints.tolist())
    if not isinstance(result, (list, tuple)) or len(result) != 7 or result[0] != 0:
        raise RuntimeError('Target inverse kinematics failed')
    q = np.asarray(result[1:], float)
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite inverse kinematics')
    fk = rpc.call('GetForwardKin', q.tolist())
    if not isinstance(fk, (list, tuple)) or len(fk) != 7 or fk[0] != 0:
        raise RuntimeError('Target forward kinematics failed')
    actual = np.asarray(fk[1:], float)
    angle_error = (actual[3:]-pose[3:]+180) % 360-180
    if (not np.isfinite(actual).all() or np.linalg.norm(actual[:3]-pose[:3]) > .5
            or np.max(abs(angle_error)) > .1):
        raise ValueError('Inverse/forward kinematics disagree')
    return {'target_pose_mm_deg': pose.tolist(), 'target_joint_deg': q.tolist(),
            'joint_delta_deg': (q-joints).tolist(), 'motion_commands_sent': 0}


def observation_state(state):
    """Encode measured joints and the gripper validity used by pi0.5."""
    joints = np.asarray(state['joints'], float)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError('Invalid measured joints')
    valid = gripper_feedback_valid(state)
    return [*np.deg2rad(joints).tolist(), state['gripper_position']/100 if valid else 0., float(valid)]


def observation_request(sample, task):
    import cv2
    state = sample['robot_state']
    rgb = np.asarray(sample['camera']['color_rgb'])
    if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected the training RGB640x360 input')
    if not isinstance(task, str) or not 1 <= len(task) <= 512:
        raise ValueError('A stage task instruction is required')
    if not math.isfinite(sample['sync_skew_s']) or abs(sample['sync_skew_s']) > .1:
        raise ValueError('Image/state pair is not synchronized')
    captured, basis = camera_clock(sample['camera']['metadata'])
    now = time.monotonic() if basis == MONOTONIC else time.time()
    if not 0 <= now-captured <= .5:
        raise ValueError('Source observation is stale')
    values = observation_state(state)
    ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('RGB encoding failed')
    return {'schema': 'fr5-pi05-request-v1', 'id': uuid.uuid4().hex,
            'state_rad_fraction_valid': values, 'task': task,
            'rgb_jpeg_base64': base64.b64encode(jpeg.tobytes()).decode(),
            'source_time': captured, 'source_clock': basis}


def decoded_action(reply, request):
    if (reply.get('schema') != 'fr5-pi05-reply-v1' or reply.get('id') != request['id']
            or reply.get('adapter_sha256') != ADAPTER_SHA
            or reply.get('action_representation') != 'absolute_joint_rad_gripper_fraction'):
        raise ValueError('Reply does not match observation/checkpoint/action contract')
    chunk = np.asarray(reply.get('actions'), float)
    if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
        raise ValueError('Expected eight finite seven-value actions')
    return [*np.rad2deg(chunk[0, :6]).tolist(), float(chunk[0, 6])*100]


def action_target(reply, request, current, phase):
    raw = decoded_action(reply, request)
    if not 0 <= raw[6] <= 100:
        raise ValueError('Predicted gripper target is outside0..100; not silently clipped')
    if not gripper_feedback_valid(current):
        raise ValueError('No valid measured gripper feedback for execution')
    if phase in ARM_PHASES:
        target = [*raw[:6], float(current['gripper_position'])]
        projection = 'keep_measured_gripper_during_arm_phase'
    elif phase in ('close', 'release'):
        target = [*current['joints'], raw[6]]
        projection = 'keep_measured_joints_during_gripper_phase'
        if (phase == 'close' and target[6] > current['gripper_position']+.5
                or phase == 'release' and target[6] < current['gripper_position']-.5):
            raise ValueError('Gripper direction contradicts the selected task phase')
    else:
        raise ValueError('Unknown task phase')
    return {'raw_target_deg_percent': raw, 'target_deg_percent': target,
            'joint_delta_deg': (np.asarray(target[:6])-current['joints']).tolist(),
            'phase_projection': projection, 'selected_action_index': 0}


def feedback_unchanged(sample, current):
    previous = sample['robot_state']
    for key in ('main_code', 'sub_code', 'EmergencyStop', 'collisionState',
                'safety_stop0_state', 'safety_stop1_state', 'gripper_fault', 'gripper_fault_id'):
        if current[key] != 0:
            raise RuntimeError('Controller fault: '+key)
    if (current['motion_done'] != 1 or current['mc_queue_len'] != 0
            or current['robot_state'] != 1 or current['rbtEnableState'] != 1
            or current['tool'] != previous['tool'] or current['user'] != previous['user']
            or np.max(abs(np.asarray(current['joints'])-previous['joints'])) > .05
            or np.max(abs(np.asarray(current['tcp'])-previous['tcp'])) > .15
            or abs(current['gripper_position']-previous['gripper_position']) > 1):
        raise RuntimeError('Robot changed during inference')


class Pi05StepBridge:
    def __init__(self, observer, policy, executor=None, *, max_inference_age_s=2.):
        if not 0 < max_inference_age_s <= 2:
            raise ValueError('Inference age must be positive and at most2s')
        self.observer, self.policy, self.executor = observer, policy, executor
        self.max_age = max_inference_age_s

    def step(self, phase, task, *, execute=False):
        # Each call samples again; the remaining predicted chunk is discarded.
        sample = deepcopy(self.observer.get_sample())
        request = observation_request(sample, task)
        reply = self.policy.predict(request)
        now = time.monotonic() if request['source_clock'] == MONOTONIC else time.time()
        if not 0 <= now-request['source_time'] <= self.max_age:
            raise TimeoutError('Observation expired during inference; no command sent')
        raw = decoded_action(reply, request)
        current = self.observer.stream.latest(max_age_s=.25)
        report = {'phase': phase, 'request_id': request['id'], 'adapter_sha256': ADAPTER_SHA,
                  'raw_target_deg_percent': raw,
                  'measured_joints_deg': list(sample['robot_state']['joints']),
                  'gripper_feedback_valid': gripper_feedback_valid(sample['robot_state']),
                  'source_time': request['source_time'], 'source_clock': request['source_clock'],
                  'inference_seconds': reply['inference_seconds'],
                  'observation_age_s': now-request['source_time'], 'execution_requested': execute,
                  'status': 'preview_only', 'task_success_claimed': False}
        try:
            feedback_unchanged(sample, current)
            proposal = action_target(reply, request, current, phase)
        except (RuntimeError, ValueError) as exc:
            if execute:
                raise
            # Preview exposes the real output and the exact rejection. It
            # does not manufacture a gripper observation or relax execution.
            report.update(status='preview_rejected_for_execution', rejection=str(exc),
                          physical_commands_sent=0)
            return report
        report['proposal'] = proposal
        if execute:
            if self.executor is None or not self.executor.armed:
                raise RuntimeError('An already-armed reviewed executor is required')
            # Existing executor rechecks current state, IK/FK, limits and the
            # actual wire write, then waits for completion. No auto-arm here.
            report['execution'] = self.executor.execute(proposal['target_deg_percent'])
            report['after'] = self.observer.get_sample()['robot_state']
            report['status'] = 'bounded_step_completed'
        return report


class SSHPolicy:
    def __init__(self, *, gpu=9):
        if type(gpu) is not int or not 0 <= gpu <= 9:
            raise ValueError('Invalid GPU index')
        remote = shlex.join([PYTHON, '-u', SERVER, '--gpu', str(gpu)])
        self.process = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=3',
            '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', '4090', remote],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.buffer = bytearray()
        self.ready = False

    def receive(self, timeout):
        deadline = time.monotonic()+timeout
        while b'\n' not in self.buffer:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('Policy server response timeout')
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError('Policy process disconnected')
            self.buffer.extend(data)
            if len(self.buffer) > 2*1024*1024:
                raise ValueError('Oversized server response')
        raw, _, tail = self.buffer.partition(b'\n')
        self.buffer = bytearray(tail)
        return json.loads(raw)

    def wait_ready(self):
        event = self.receive(30)
        if event.get('status') == 'ready':
            if event.get('adapter_sha256') != ADAPTER_SHA:
                raise ValueError('Unexpected loaded model')
            self.ready = True
        return event

    def predict(self, request):
        if not self.ready:
            raise RuntimeError('Policy not ready')
        raw = (json.dumps(request, allow_nan=False)+'\n').encode()
        if len(raw) > 2*1024*1024:
            raise ValueError('Request exceeds protocol limit')
        view = memoryview(raw)
        while view:
            count = self.process.stdin.write(view)
            if not count:
                raise RuntimeError('Policy pipe write failed')
            view = view[count:]
        reply = self.receive(10)
        if reply.get('status') == 'rejected':
            raise RuntimeError('Policy rejected observation: '+str(reply.get('error')))
        return reply

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=sorted(ARM_PHASES | {'close', 'release'}), required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--steps', type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    from .configuration_fairino import FairinoFR5Config
    from .robot_fairino import FairinoFR5
    from lerobot_camera_gemini import GeminiBridgeConfig
    observer = FairinoFR5(FairinoFR5Config(camera=GeminiBridgeConfig(
        url='http://127.0.0.1:8765/snapshot_fast.npz', fps=6)))
    observer.connect()
    policy = SSHPolicy()
    try:
        deadline = time.monotonic()+600
        while not policy.ready:
            if time.monotonic() > deadline:
                raise TimeoutError('Model loading exceeded600s')
            try:
                print(json.dumps(policy.wait_ready()), flush=True)
            except TimeoutError:
                print(json.dumps({'status': 'loading', 'robot_commands_sent': 0}), flush=True)
        bridge = Pi05StepBridge(observer, policy)
        for _ in range(args.steps):
            print(json.dumps(bridge.step(args.phase, args.task), allow_nan=False), flush=True)
    finally:
        policy.close()
        observer.disconnect()


if __name__ == '__main__':
    main()
