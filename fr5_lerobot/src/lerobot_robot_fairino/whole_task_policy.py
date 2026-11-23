"""Versioned SSH policy client. Loading/inference never connects to a robot."""
from __future__ import annotations

import json
import os
from pathlib import PurePosixPath
import select
import shlex
import subprocess
import time

import numpy as np

from .pi05_bridge import observation_request
from .task_contract import NAME, CONTRACT, CHUNK, ACTION_DIM, state_vector

SERVER_ROOT = PurePosixPath('/home/data4/hts/demo1/fr5-learning')
SERVER_PYTHON = str(SERVER_ROOT/'.venv/bin/python')


def data4_path(value):
    path = PurePosixPath(str(value))
    if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(SERVER_ROOT):
        raise ValueError('Remote model/code path must be inside data4/fr5-learning')
    return str(path)


def make_request(sample, task, previous_target, completed_events, previous_dwell_s, initial_time):
    request = observation_request(sample, task)
    if request['source_clock'] != 'host_monotonic':
        raise ValueError('Whole-task execution requires the shared monotonic source clock')
    request['schema'] = 'fr5-pi05-request-v2'
    request['policy_contract'] = NAME
    request['state'] = state_vector(request.pop('state_rad_fraction_valid'), previous_target,
        completed_events, previous_dwell_s, request['source_time']-initial_time)
    return request


def validate_reply(reply, request, checkpoint_sha):
    if (reply.get('schema') != 'fr5-pi05-reply-v2' or reply.get('id') != request['id']
            or reply.get('checkpoint_sha256') != checkpoint_sha or reply.get('policy_contract') != NAME):
        raise ValueError('Reply observation, checkpoint or policy contract mismatch')
    actions = np.asarray(reply.get('actions'), dtype=float)
    if actions.shape != (CHUNK, ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError('Malformed model action chunk')
    return actions[0].tolist()


class WholeTaskPolicy:
    def __init__(self, checkpoint, server_source=None, *, gpu=None, expected_sha=None, process_factory=subprocess.Popen):
        self.checkpoint = data4_path(checkpoint)
        self.server_source = data4_path(server_source) if server_source else None
        if gpu is not None and (type(gpu) is not int or gpu < 0):
            raise ValueError('GPU index must be a nonnegative integer')
        server = (PurePosixPath(self.server_source)/'fr5_lerobot/training/serve_whole_task_pi05.py'
                  if self.server_source else PurePosixPath(self.checkpoint)/'serve_whole_task_pi05.py')
        command = [SERVER_PYTHON, '-u', str(server),
                   '--checkpoint', self.checkpoint]
        if gpu is not None:
            command += ['--gpu', str(gpu)]
        self.process = process_factory(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
            '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', '4090', shlex.join(command)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.buffer = bytearray()
        self.ready, self.checkpoint_sha = False, None
        self.training_data_kind = None
        self.expected_sha = expected_sha

    def receive(self, timeout):
        deadline = time.monotonic()+timeout
        while b'\n' not in self.buffer:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('Policy response timeout')
            value = os.read(self.process.stdout.fileno(), 65536)
            if not value:
                raise RuntimeError('Policy server disconnected')
            self.buffer.extend(value)
            if len(self.buffer) > 2*1024*1024:
                raise ValueError('Policy response exceeds protocol size')
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        return json.loads(line)

    def wait_ready(self, timeout=30):
        event = self.receive(timeout)
        if event.get('status') == 'ready':
            digest = event.get('checkpoint_sha256')
            if (event.get('policy_contract') != CONTRACT or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)
                    or (self.expected_sha is not None and self.expected_sha != digest)):
                raise ValueError('Loaded checkpoint or contract differs from requested model')
            self.checkpoint_sha, self.ready = digest, True
            self.training_data_kind = event.get('training_data_kind')
        elif event.get('status') in ('error', 'rejected'):
            raise RuntimeError('Model load failed: '+str(event.get('error')))
        return event

    def predict(self, request):
        if not self.ready:
            raise RuntimeError('Policy has not loaded')
        payload = (json.dumps(request, allow_nan=False)+'\n').encode()
        if len(payload) > 2*1024*1024:
            raise ValueError('Observation exceeds protocol size')
        view = memoryview(payload)
        while view:
            count = self.process.stdin.write(view)
            if not count:
                raise RuntimeError('Policy input closed')
            view = view[count:]
        result = self.receive(30)
        if result.get('status') == 'rejected':
            raise RuntimeError('Model rejected input: '+str(result.get('error')))
        validate_reply(result, request, self.checkpoint_sha)
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()

