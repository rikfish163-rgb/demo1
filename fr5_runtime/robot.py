"""J1 state reads and XML-RPC transport for FR5 software v3.8.4.2."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import ctypes
from pathlib import Path
import socket
import struct
import threading
import time
import xmlrpc.client

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = '192.168.58.2'
_state_source_lock = threading.RLock()
_shared_state_source = None


class Transport(xmlrpc.client.Transport):
    def __init__(self, timeout=3):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self.timeout
        return conn


def rpc(timeout=3):
    return xmlrpc.client.ServerProxy(f'http://{HOST}:20003', transport=Transport(timeout))


def state_type():
    source = ROOT / 'hardware/sdk/Robot.py'
    names = {'ROBOT_AUX_STATE', 'EXT_AXIS_STATUS', 'WELDING_BREAKOFF_STATE', 'RobotStatePkg'}
    classes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
    namespace = {**vars(ctypes), 'ctypes': ctypes}
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<official-state-structures>', 'exec'), namespace)
    return namespace['RobotStatePkg']


STATE = state_type()


@contextmanager
def use_shared_state_stream(stream, *, max_age_s=0.25):
    """Let existing executors share the collector's already-connected receiver.

    Scoped to this process, including worker threads. A failed/stale shared
    receiver raises; it never falls back to opening another controller socket.
    The caller owns connect/disconnect and must join workers before exit.
    This changes only observation routing, never the execution gates.
    """
    global _shared_state_source
    if (getattr(stream, 'host', None) != HOST or
            getattr(stream, 'port', None) != 20004):
        raise ValueError('Shared stream must refer to the same J1 state endpoint')
    if isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float)) or not 0 < max_age_s <= .5:
        raise ValueError('Shared state freshness must be positive and <=0.5s')
    with _state_source_lock:
        if _shared_state_source is not None:
            raise RuntimeError('Another shared state scope is already active')
        stream.latest(max_age_s=max_age_s)
        _shared_state_source = (stream, max_age_s)
    try:
        yield
    finally:
        with _state_source_lock:
            _shared_state_source = None


def read_state():
    with _state_source_lock:
        if _shared_state_source is not None:
            stream, max_age_s = _shared_state_source
            return deepcopy(stream.latest(max_age_s=max_age_s))
        return _read_state_once()


def _read_state_once():
    with socket.create_connection((HOST, 20004), timeout=2) as conn:
        conn.settimeout(2)
        buffer = bytearray()
        while len(buffer) < 65536:
            received = conn.recv(4096)
            if not received:
                raise RuntimeError('State stream disconnected')
            buffer.extend(received)
            index = buffer.find(b'\x5a\x5a')
            if index < 0 or len(buffer) < index + 5:
                continue
            count = struct.unpack_from('<H', buffer, index+3)[0] + 7
            if len(buffer) < index + count:
                continue
            packet = bytes(buffer[index:index+count])
            if sum(packet[:-2]) % 65536 != int.from_bytes(packet[-2:], 'little'):
                del buffer[:index+2]
                continue
            # 3.8.4.2 has two more trailing bytes than the 3.8.4 structure;
            # fields used below precede this differing tail and are in bounds.
            if len(packet) < ctypes.sizeof(STATE):
                raise RuntimeError('State packet shorter than verified structure')
            pkg = STATE.from_buffer_copy(packet)
            scalar = ['program_state', 'robot_state', 'main_code', 'sub_code', 'robot_mode',
                      'tool', 'user', 'EmergencyStop', 'motion_done', 'mc_queue_len',
                      'collisionState', 'safety_stop0_state', 'safety_stop1_state',
                      'gripper_fault', 'gripper_position', 'rbtEnableState',
                      'gripper_motiondone', 'gripper_active', 'gripper_speed',
                      'gripper_current', 'gripper_fault_id']
            result = {name: getattr(pkg, name) for name in scalar}
            result.update(joints=list(pkg.jt_cur_pos), tcp=list(pkg.tl_cur_pos),
                          flange=list(pkg.flange_cur_pos), tcp_speed=list(pkg.actual_TCP_CmpSpeed),
                          joint_speeds_deg_s=list(pkg.actual_qd),
                          timestamp=time.time())
            return result
        raise RuntimeError('No valid state packet')


def require_ready(state, stationary=True):
    if any(state[k] != 0 for k in ['main_code', 'sub_code', 'EmergencyStop',
                                   'collisionState', 'safety_stop0_state', 'safety_stop1_state']):
        raise RuntimeError('Fault, emergency stop, collision or safety stop active')
    allowed_program_states = [1] if stationary else [1,2]
    if state['program_state'] not in allowed_program_states or state['rbtEnableState'] != 1:
        raise RuntimeError('Program is not stopped or robot is not enabled')
    if state['tool'] != 0 or state['user'] != 0:
        raise RuntimeError('Expected tool0 and user0')
    if np.linalg.norm(np.array(state['tcp'][:3])-state['flange'][:3]) > 1:
        raise RuntimeError('Tool0 TCP does not match flange')
    if stationary and (state['robot_state'] != 1 or state['motion_done'] != 1 or state['mc_queue_len'] != 0):
        raise RuntimeError('Robot is not stationary with an empty command queue')


class SingleAttemptTransport(Transport):
    def request(self, host, handler, request_body, verbose=False):
        # xmlrpc.client's default transport can retry a disconnected request.
        # A motion request must never be replayed after an ambiguous response.
        return self.single_request(host, handler, request_body, verbose)


def controller():
    return xmlrpc.client.ServerProxy(
        f'http://{HOST}:20003', transport=SingleAttemptTransport(2))
