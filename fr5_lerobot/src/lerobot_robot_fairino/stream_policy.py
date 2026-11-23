"""Persistent remote inference client for the versioned seven-output policy."""
from __future__ import annotations

from pathlib import PurePosixPath
import json
import shlex
import subprocess

import numpy as np

from .pi05_bridge import observation_request
from .stream_contract import NAME, CHUNK, ACTION_DIM, CUP_RELATIVE_WIRE, ABSOLUTE_JOINT_WIRE
from .whole_task_policy import WholeTaskPolicy, data4_path, SERVER_PYTHON
from fr5_lerobot.training.stream_observation_history import (
    CUP_HISTORY_PROFILES, CUP_TCP_HISTORY_PROFILES, CUP_TIMING_PROFILES, CUP_TRACKING_PROFILES,
    HISTORY_PROFILES, PROFILES, cup_timing_input, cup_tracking_input, initial_cup_pixel,
)


def make_request(sample, task, *, observation_context_profile='none'):
    request = observation_request(sample, task)
    if request['source_clock'] != 'host_monotonic':
        raise ValueError('Live stream requires the shared host monotonic clock')
    request.update(schema='fr5-stream-request-v1', policy_contract=NAME,
                   state=request.pop('state_rad_fraction_valid'))
    if observation_context_profile in HISTORY_PROFILES:
        request['observation_context_profile'] = observation_context_profile
        request['observation_history'] = sample.get('observation_history', [None]*6)
    elif observation_context_profile != 'none':
        raise ValueError('Unknown observation context profile')
    if observation_context_profile in CUP_HISTORY_PROFILES:
        request['initial_cup_uv'] = initial_cup_pixel(sample.get('initial_cup_uv'))
    if observation_context_profile in CUP_TCP_HISTORY_PROFILES:
        from .cup_relative_runtime import cup_relative_state
        cup = sample['initial_cup']
        request['cup_base_xyz_mm'] = cup['base_xyz_mm']
        request['cup_relative_state'] = cup_relative_state(sample, cup)
    if observation_context_profile in CUP_TRACKING_PROFILES:
        tracking = sample['cup_tracking']
        request['cup_tracking'] = cup_tracking_input({k: tracking[k] for k in ('rim_minus_jaw_mm', 'native_contact')})
    if observation_context_profile in CUP_TIMING_PROFILES:
        request['cup_timing'] = cup_timing_input(sample['cup_timing']['state'])
    return request


def validate_reply(reply, request, checkpoint_sha):
    context = request.get('observation_context_profile', 'none')
    wire_mode = CUP_RELATIVE_WIRE if context in CUP_TCP_HISTORY_PROFILES else ABSOLUTE_JOINT_WIRE
    if (reply.get('schema') != 'fr5-stream-reply-v1' or reply.get('id') != request['id']
            or reply.get('checkpoint_sha256') != checkpoint_sha
            or reply.get('policy_contract') != NAME
            or reply.get('observation_context_profile', 'none') != context
            or reply.get('wire_action_representation', ABSOLUTE_JOINT_WIRE) != wire_mode):
        raise ValueError('Stream reply observation/checkpoint/contract mismatch')
    values = np.asarray(reply.get('actions'), dtype=float)
    if values.shape != (CHUNK, ACTION_DIM) or not np.isfinite(values).all():
        raise ValueError('Expected a finite 50 by 7 action chunk')
    return values.tolist()


class StreamPolicy(WholeTaskPolicy):
    """Reuse only framed transport; legacy ten-output weights are rejected."""

    def __init__(self, checkpoint, server_source=None, *, gpu=None, expected_sha=None,
                 process_factory=subprocess.Popen):
        self.checkpoint = data4_path(checkpoint)
        self.server_source = data4_path(server_source) if server_source else None
        if gpu is not None and (type(gpu) is not int or gpu < 0):
            raise ValueError('GPU index must be a nonnegative integer')
        server = (PurePosixPath(self.server_source)/'fr5_lerobot/training/serve_stream_pi05.py'
                  if self.server_source else PurePosixPath(self.checkpoint)/'serve_stream_pi05.py')
        command = [SERVER_PYTHON, '-u', str(server), '--checkpoint', self.checkpoint]
        if gpu is not None:
            command += ['--gpu', str(gpu)]
        self.process = process_factory(['ssh', '-T', '-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
            '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', '4090', shlex.join(command)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.buffer = bytearray()
        self.ready, self.checkpoint_sha = False, None
        self.n_action_steps = CHUNK
        self.training_data_kind = None
        self.observation_context_profile = 'none'
        self.expected_sha = expected_sha

    def wait_ready(self, timeout=30):
        event = self.receive(timeout)
        if event.get('status') == 'ready':
            digest = event.get('checkpoint_sha256')
            steps = event.get('n_action_steps', CHUNK)
            context = event.get('observation_context_profile', 'none')
            wire_mode = CUP_RELATIVE_WIRE if context in CUP_TCP_HISTORY_PROFILES else ABSOLUTE_JOINT_WIRE
            if (event.get('schema') != 'fr5-stream-ready-v1' or event.get('policy_contract') != NAME
                    or event.get('state_shape') != [8] or event.get('action_shape') != [CHUNK, ACTION_DIM]
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)
                    or type(steps) is not int or not 1 <= steps <= CHUNK
                    or context not in PROFILES
                    or event.get('wire_action_representation', ABSOLUTE_JOINT_WIRE) != wire_mode
                    or (self.expected_sha is not None and digest != self.expected_sha)):
                raise ValueError('Loaded stream checkpoint differs from requested contract')
            self.ready, self.checkpoint_sha = True, digest
            self.n_action_steps = steps
            self.training_data_kind = event.get('training_data_kind')
            self.observation_context_profile = context
        elif event.get('status') in ('error', 'rejected'):
            raise RuntimeError('Stream model load failed: '+str(event.get('error')))
        return event

    def predict(self, request):
        if not self.ready:
            raise RuntimeError('Stream model has not loaded')
        if request.get('observation_context_profile', 'none') != self.observation_context_profile:
            raise ValueError('Observation context differs from loaded model')
        payload = (json.dumps(request, allow_nan=False)+'\n').encode()
        if len(payload) > 2*1024*1024:
            raise ValueError('Observation exceeds protocol limit')
        view = memoryview(payload)
        while view:
            count = self.process.stdin.write(view)
            if not count:
                raise RuntimeError('Stream model input closed')
            view = view[count:]
        reply = self.receive(30)
        if reply.get('status') in ('error', 'rejected'):
            raise RuntimeError('Stream model rejected input: '+str(reply.get('error')))
        validate_reply(reply, request, self.checkpoint_sha)
        return reply
