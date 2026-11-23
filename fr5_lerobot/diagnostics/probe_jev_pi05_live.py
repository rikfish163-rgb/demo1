#!/usr/bin/env python3
"""Record read-only FR5 observations and complete remote pi0.5 predictions."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import select
import shlex
import subprocess
import threading
import time

from fr5_runtime.system import robot_lease, ssh_args
from fr5_lerobot.training.stream_observation_history import (
    NONE_PROFILE, PHYSICAL_HISTORY_PROFILE, select_observation_history,
)
from lerobot_camera_gemini import GeminiBridgeConfig
from lerobot_camera_gemini.timing import MONOTONIC, sample_clock
from lerobot_robot_fairino.configuration_fairino import FairinoFR5Config
from lerobot_robot_fairino.continuous_capture import _wait_until_monotonic
from lerobot_robot_fairino.feedback import gripper_feedback_valid
from lerobot_robot_fairino.pi05_bridge import observation_state
from lerobot_robot_fairino.robot_fairino import FairinoFR5
from lerobot_robot_fairino.stream_contract import FPS
from lerobot_robot_fairino.stream_policy import StreamPolicy, make_request
from lerobot_robot_fairino.whole_task_policy import SERVER_PYTHON, data4_path


PROFILES = (NONE_PROFILE, PHYSICAL_HISTORY_PROFILE)
HISTORY_TIMEOUT_S = 20.0
MAX_CAPTURE_GAP_S = max(.5, 2.5 / FPS)

_REMOTE_ARCHIVE = r'''
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('xb') as output:
    print(json.dumps({'status': 'ready'}), flush=True)
    count = 0
    for line in sys.stdin.buffer:
        output.write(line)
        output.flush()
        count += 1
    os.fsync(output.fileno())
print(json.dumps({'status': 'complete', 'records': count}), flush=True)
'''


class RemoteArchive:
    """Stream JSON lines directly to a new data4 file over SSH."""

    def __init__(self, path):
        self.path = data4_path(path)
        command = shlex.join([SERVER_PYTHON, '-u', '-c', _REMOTE_ARCHIVE, self.path])
        self.process = subprocess.Popen(ssh_args(command), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.count = 0
        try:
            if not select.select([self.process.stdout], [], [], 15)[0]:
                raise TimeoutError('Remote archive did not become ready')
            line = self.process.stdout.readline()
            if not line or json.loads(line).get('status') != 'ready':
                raise RuntimeError('Remote archive could not create a new output file')
        except BaseException:
            self.process.terminate()
            self.process.communicate()
            raise

    def write(self, record):
        payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode()
        view = memoryview(payload)
        while view:
            count = self.process.stdin.write(view)
            if not count:
                raise RuntimeError('Remote archive input closed')
            view = view[count:]
        self.count += 1

    def close(self):
        self.process.stdin.close()
        self.process.stdin = None
        try:
            stdout, stderr = self.process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.communicate()
            raise TimeoutError('Remote archive did not finish')
        if self.process.returncode:
            raise RuntimeError('Remote archive failed: ' + stderr.decode(errors='replace').strip())
        receipt = json.loads(stdout)
        if receipt != {'status': 'complete', 'records': self.count}:
            raise RuntimeError('Remote archive record count differs')


class LiveSamples:
    """Keep measured camera-clock state history while inference is running."""

    def __init__(self, robot):
        self.robot = robot
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.history = deque(maxlen=128)
        self.latest = None
        self.error = None
        self.thread = threading.Thread(target=self._run, name='fr5-live-probe-observations', daemon=True)

    def start(self):
        self.thread.start()

    def _accept(self, sample):
        captured, _, basis = sample_clock(sample)
        if basis != MONOTONIC:
            raise ValueError('Live observations require the host monotonic camera clock')
        if not 0 <= time.monotonic() - captured <= .5:
            raise TimeoutError('Camera observation is stale')
        skew = sample['sync_skew_s']
        if not math.isfinite(skew) or abs(skew) > .1:
            raise ValueError('Camera and robot observation are not synchronized')
        state = observation_state(sample['robot_state'])
        with self.condition:
            if self.history:
                gap = captured - self.history[-1][0]
                if not 0 < gap <= MAX_CAPTURE_GAP_S:
                    raise ValueError('Camera history has a duplicate, backward timestamp or capture gap')
            self.history.append((captured, state))
            self.latest = sample
            self.condition.notify_all()

    def _run(self):
        try:
            while not self.stopped.is_set():
                tick = time.monotonic()
                self._accept(self.robot.get_sample())
                _wait_until_monotonic(self.stopped, tick + 1 / FPS)
        except BaseException as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def wait(self, profile, *, after=None, timeout=HISTORY_TIMEOUT_S):
        if profile not in PROFILES:
            raise ValueError('Expected a current-state or physical-history checkpoint')
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError('Read-only observation failed') from self.error
                if self.latest is not None:
                    captured = self.history[-1][0]
                    if after is None or captured > after:
                        value = dict(self.latest)
                        if profile == NONE_PROFILE:
                            return value
                        selected = select_observation_history(self.history, captured)
                        if all(item is not None for item in selected.values()):
                            value['observation_history'] = [
                                {'age_s': item['age_s'], 'state': item['state']}
                                for item in selected.values()
                            ]
                            return value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('No fresh observation with the required complete history')
                self.condition.wait(remaining)

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2 * self.robot.config.camera.timeout_s + 5)
        if self.thread.is_alive():
            raise TimeoutError('Read-only observation worker did not stop')
        if self.error is not None:
            raise RuntimeError('Read-only observation failed') from self.error


def feedback_after(stream, timestamp, *, timeout=2.0):
    """Return a real state received after the inference reply."""
    deadline = time.monotonic() + timeout
    while True:
        state = stream.latest(max_age_s=.25)
        if state['received_monotonic'] > timestamp:
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError('No fresh robot feedback after the reply')
        time.sleep(.01)


def _predict_sample(policy, robot, sample, task):
    observation_ready = time.monotonic()
    request = make_request(sample, task,
                           observation_context_profile=policy.observation_context_profile)
    before = robot.stream.latest(max_age_s=.25)
    request_started = time.monotonic()
    reply = policy.predict(request)
    reply_received = time.monotonic()
    duration = reply.get('inference_seconds')
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError('The server did not report a valid inference duration')
    after_state = feedback_after(robot.stream, reply_received)
    return {
        'type': 'prediction',
        'request': request,
        'reply': reply,
        'camera_metadata': sample['camera']['metadata'],
        'sync_skew_s': sample['sync_skew_s'],
        'paired_robot_feedback': sample['robot_state'],
        'robot_feedback_before': before,
        'robot_feedback_after': after_state,
        'paired_gripper_feedback_valid': gripper_feedback_valid(sample['robot_state']),
        'before_gripper_feedback_valid': gripper_feedback_valid(before),
        'after_gripper_feedback_valid': gripper_feedback_valid(after_state),
        'observation_ready_monotonic': observation_ready,
        'request_started_monotonic': request_started,
        'reply_received_monotonic': reply_received,
        'request_roundtrip_s': reply_received - request_started,
        'source_to_reply_s': reply_received - request['source_time'],
        'server_inference_seconds': duration,
    }


def sample_prediction(policy, robot, observations, task, *, after=None):
    sample = observations.wait(policy.observation_context_profile, after=after)
    return _predict_sample(policy, robot, sample, task)


def sample_predictions(policies, robot, observations, task, *, after=None):
    """Compare two policies concurrently using the same fresh measured frame."""
    profiles = [policy.observation_context_profile for policy in policies.values()]
    if len(profiles) != 2 or any(profile not in PROFILES for profile in profiles):
        raise ValueError('Expected two current-state or physical-history policies')
    profile = PHYSICAL_HISTORY_PROFILE if PHYSICAL_HISTORY_PROFILE in profiles else NONE_PROFILE
    sample = observations.wait(profile, after=after)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(_predict_sample, policy, robot, sample, task)
                   for name, policy in policies.items()}
        return {name: future.result() for name, future in futures.items()}


def run(policy, robot, archive, task, count):
    if policy.observation_context_profile not in PROFILES or policy.n_action_steps != 10:
        raise ValueError('Expected a current/history stream checkpoint with 10 action steps')
    archive.write({'type': 'session', 'schema': 'fr5-pi05-live-probe-v1',
                   'checkpoint': policy.checkpoint, 'checkpoint_sha256': policy.checkpoint_sha,
                   'observation_context_profile': policy.observation_context_profile,
                   'n_action_steps': policy.n_action_steps, 'task': task,
                   'requested_predictions': count, 'capture_fps': FPS,
                   'robot_commands_sent': 0})
    observations = None
    try:
        with robot_lease():
            robot.connect()
            try:
                observations = LiveSamples(robot)
                observations.start()
                after = None
                for index in range(count):
                    row = sample_prediction(policy, robot, observations, task, after=after)
                    row['index'] = index
                    archive.write(row)
                    after = row['reply_received_monotonic']
            finally:
                try:
                    if observations is not None:
                        observations.close()
                finally:
                    robot.disconnect()
    except BaseException as exc:
        archive.write({'type': 'error', 'error': f'{type(exc).__name__}: {exc}',
                       'time_monotonic': time.monotonic(), 'robot_commands_sent': 0})
        raise
    archive.write({'type': 'complete', 'predictions': count, 'robot_commands_sent': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path on data4')
    parser.add_argument('--server-source', help='Optional published source root on data4')
    parser.add_argument('--task', required=True, help='Complete task instruction')
    parser.add_argument('--output', required=True, help='New JSONL path on data4')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--gpu', type=int, help='Optional remote GPU index')
    args = parser.parse_args()
    if args.samples < 1 or not args.task.strip():
        raise ValueError('A positive sample count and a nonempty task are required')
    data4_path(args.output)
    policy = StreamPolicy(args.checkpoint, args.server_source, gpu=args.gpu)
    try:
        deadline = time.monotonic() + 300
        while not policy.ready:
            try:
                policy.wait_ready(timeout=min(30, max(.01, deadline - time.monotonic())))
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise
        archive = RemoteArchive(args.output)
        try:
            robot = FairinoFR5(FairinoFR5Config(camera=GeminiBridgeConfig(
                url='http://127.0.0.1:8765/snapshot_fast.npz', fps=FPS)))
            run(policy, robot, archive, args.task, args.samples)
        finally:
            archive.close()
    finally:
        policy.close()
    print(json.dumps({'status': 'complete', 'output': args.output, 'predictions': args.samples,
                      'robot_commands_sent': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
