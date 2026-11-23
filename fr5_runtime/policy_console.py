"""Local policy workbench alongside the existing collection workbench.

The robot is controlled only by the existing fr5-system runtime. Each browser
start creates a detached worker; the server never takes over another run.
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import math
import os
import re
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from fr5_runtime import system

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'fr5_lerobot/src/lerobot_robot_fairino/web/policy.html'
PYTHON = '/home/dianrobot/anaconda3/envs/lerobot/bin/python'
STATE = Path.home() / '.local/state/fr5-policy-console'
MONITOR = 'http://127.0.0.1:8765'
DEFAULT_TASK = '抓取桌面上的透明杯子，在空中维持5秒后放下'


def read_object(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def write_state(name, value):
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / (system.simple_name(name) + '.json')
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def process_args(pid):
    try:
        path = Path('/proc') / str(int(pid))
        if path.stat().st_uid != os.getuid():
            return []
        return [item.decode('utf-8', 'replace') for item in (path / 'cmdline').read_bytes().split(b'\0') if item]
    except (OSError, ValueError, TypeError):
        return []


def flag_value(args, flag):
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def runtime_released(pid, episode):
    try:
        system.simple_name(episode)
    except ValueError:
        return False
    marker=read_object(system.LOCAL_STATE/('released-'+episode+'.json'))
    return (marker.get('pid')==pid and marker.get('episode')==episode
            and marker.get('robot_lease_released') is True)


def active_runtimes():
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        args = process_args(entry.name)
        positions = [i for i, arg in enumerate(args) if arg == 'fr5_runtime.system']
        if not positions or positions[0] + 1 >= len(args) or args[positions[0] + 1] != 'run' or '--execute' not in args:
            continue
        name = flag_value(args, '--name')
        try:
            system.simple_name(name)
        except ValueError:
            continue
        if runtime_released(int(entry.name),name):
            continue
        result.append({'pid': int(entry.name), 'episode': name, 'checkpoint': flag_value(args, '--checkpoint'),
                       'gpu': flag_value(args, '--gpu')})
    return result


def worker_alive(pid, episode):
    args = process_args(pid)
    return ('fr5_runtime.policy_console' in args and flag_value(args, '--worker') == episode)


def robot_busy():
    path = system.LOCAL_STATE / 'j1.lock'
    if not path.exists():
        return False
    try:
        with path.open('r') as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            finally:
                try:
                    fcntl.flock(file, fcntl.LOCK_UN)
                except OSError:
                    pass
    except OSError:
        return None
    return False


def monitor_status():
    try:
        with urllib.request.urlopen(MONITOR + '/status', timeout=2) as response:
            value = json.load(response)
        return {'camera': value.get('camera', {}), 'robot': value.get('robot', {})}
    except (OSError, ValueError):
        return {'camera': {'ok': False}, 'robot': {'ok': False}}


def checkpoint():
    value = system.configuration().get('last_checkpoint')
    if not isinstance(value, str):
        return None
    path = Path(value)
    base = Path(system.REMOTE) / 'checkpoints'
    if path.parent != base:
        # Stream training saves an immutable bundle for each cumulative step.
        if (path.parent.parent != base or not re.fullmatch(r'step-[0-9]{6,}', path.name)):
            return None
    try:
        system.simple_name(path.name)
        if path.parent != base:
            system.simple_name(path.parent.name)
    except ValueError:
        return None
    return str(path)


def selected_checkpoint_contract(model, episode=None):
    """Describe the selected bundle, rather than another experiment's model."""
    known=system.configuration().get('checkpoint_contracts',{})
    value=known.get(model) if isinstance(known,dict) else None
    if isinstance(value,dict) and isinstance(value.get('name'),str):
        return value['name']
    if episode and episode.get('checkpoint')==model:
        return episode.get('policy_contract')
    return None


def episode_view(name):
    if not name:
        return None
    try:
        system.simple_name(name)
    except ValueError:
        return None
    directory = system.SPOOL / name
    scope = read_object(directory / 'scope.json')
    if scope and scope.get('dataset_role') != 'policy_evaluation':
        return None
    files = sorted(directory.glob('chunk-*.json')) if directory.is_dir() else []
    latest_targets = {}
    target_path = directory / 'submitted-targets.jsonl'
    if target_path.exists():
        with target_path.open('rb') as data:
            data.seek(max(0, target_path.stat().st_size-256*1024))
            for line in data:
                try:
                    item = json.loads(line)
                    latest_targets[item['request_id']] = item
                except (ValueError, KeyError):
                    continue
    events = []
    for path in files[-12:]:
        value = read_object(path)
        request, reply = value.get('request') or {}, value.get('reply') or {}
        submitted = latest_targets.get(request.get('id'), {})
        decoded = submitted.get('decoded') or {}
        state = request.get('state') or []
        events.append({'file': path.name, 'event_kind': 'stream',
            'status': 'submitted' if submitted else value.get('status'),
            'joint_degrees': decoded.get('joint_degrees'),
            'gripper_target': decoded.get('gripper_integer_percent'),
            'inference_seconds': reply.get('inference_seconds'),
            'source_age_at_reply_s': value.get('source_age_at_reply_s'),
            'action_index': submitted.get('action_index'), 'dwell_seconds': None,
            'observed_gripper': round(state[6]*100, 1) if len(state)==8 and state[7]==1 else None,
            'modified_unix':path.stat().st_mtime})
    summary = read_object(directory / 'summary.json')
    receipt = read_object(directory / 'stream-closed.json')
    return {'name': name, 'exists': directory.is_dir(), 'task': scope.get('task_instruction'),
            'checkpoint': scope.get('checkpoint'), 'checkpoint_sha256': scope.get('checkpoint_sha256'),
            'runtime_llm': scope.get('runtime_llm'), 'events_total': len(files), 'events': events,
            'policy_contract': (scope.get('policy_contract') or {}).get('name'),
            'summary': {key: summary.get(key) for key in ('outcome', 'failed', 'commands',
                        'completed_policy_events', 'termination_source', 'task_success_verified')} if summary else None,
            'recorded_frames': receipt.get('accepted_frames')}


def latest_policy_episode():
    if not system.SPOOL.is_dir():
        return None
    scopes = sorted(system.SPOOL.glob('*/scope.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in scopes:
        if read_object(path).get('dataset_role') == 'policy_evaluation':
            try:
                return system.simple_name(path.parent.name)
            except ValueError:
                continue
    return None


def worker(name, model, task):
    """Drain runtime output into a small state file independently of the UI."""
    name = system.simple_name(name)
    state = {'episode': name, 'worker_pid': os.getpid(), 'status': 'loading',
             'checkpoint': model, 'task': task, 'last_message': '', 'started_at': time.time()}
    command = [PYTHON, '-u', '-m', 'fr5_runtime.system', 'run', '--checkpoint', model,
               '--task', task, '--name', name, '--execute']
    process = None
    stop_requested = False

    def stop(*_):
        nonlocal stop_requested
        stop_requested = True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    write_state(name, state)
    if stop_requested:
        state.update(status='stopped', stop_requested=True, finished_at=time.time())
        write_state(name, state)
        return
    try:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
    except OSError as exc:
        state.update(status='failed', last_message=str(exc), finished_at=time.time())
        write_state(name, state)
        raise
    state['child_pid'] = process.pid
    if stop_requested:
        state['status'] = 'stopping'
        process.send_signal(signal.SIGINT)
    write_state(name, state)
    recent = deque(maxlen=6)
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except ValueError:
                recent.append(line[-350:])
                state['last_message'] = ' · '.join(recent)[-1200:]
            else:
                kind = value.get('status') if isinstance(value, dict) else None
                if stop_requested:
                    state['status'] = 'stopping'
                if kind in ('ready', 'policy_event_completed', 'policy_stream_progress') and state['status'] != 'stopping':
                    state['status'] = 'running'
                elif kind == 'model_loading' and state['status'] != 'stopping':
                    state['status'] = 'loading'
                if kind:
                    state['last_event'] = {k: value.get(k) for k in ('status', 'record', 'event_index')}
            write_state(name, state)
    finally:
        process.stdout.close()
        state.update(status='finished', exit_code=process.wait(), stop_requested=stop_requested,
                     finished_at=time.time())
        write_state(name, state)


class PolicyConsole:
    def __init__(self):
        self.csrf = secrets.token_urlsafe(24)
        self.lock = threading.RLock()
        self.last_started = None
        self.launching = None
        self.replays = {}
        self.reviews = {}

    def current_worker(self):
        if self.launching and self.launching.poll() is None and self.last_started:
            item = read_object(STATE / (self.last_started + '.json'))
            if not runtime_released(item.get('child_pid'),self.last_started):
                return item or {'episode': self.last_started, 'worker_pid': self.launching.pid, 'status': 'loading'}
        candidates = sorted(STATE.glob('pi05-ui-*.json'), key=lambda p: p.stat().st_mtime, reverse=True) if STATE.is_dir() else []
        for path in candidates[:20]:
            item = read_object(path)
            if (item.get('episode') and worker_alive(item.get('worker_pid'), item['episode'])
                    and not runtime_released(item.get('child_pid'),item['episode'])):
                return item
        return None

    def status(self):
        current = self.current_worker()
        active = active_runtimes()
        active_run = active[0] if active else None
        owned = bool(current and (not active_run or current.get('child_pid') == active_run.get('pid')))
        name = (active_run or {}).get('episode') or (current or {}).get('episode') or self.last_started
        if not name:
            name = latest_policy_episode() or system.configuration().get('last_policy_episode')
        view = episode_view(name)
        last_worker = read_object(STATE / (name + '.json')) if name and name.startswith('pi05-ui-') else {}
        if current and current.get('status') == 'stopping':
            phase = 'stopping'
        elif active_run and view and view.get('summary'):
            phase = 'saving'
        elif current or active_run:
            phase = 'running' if view and view.get('events_total') else 'loading'
        elif view and view.get('summary'):
            phase = 'finished'
        elif last_worker.get('status') == 'stopped' or (last_worker.get('status') == 'finished' and last_worker.get('stop_requested')):
            phase = 'finished'
        elif last_worker.get('status') == 'failed' or (last_worker.get('status') == 'finished' and last_worker.get('exit_code') != 0):
            phase = 'failed'
        else:
            phase = 'idle'
        selected_model = checkpoint()
        validation=system.configuration().get('checkpoint_validation')
        if not isinstance(validation,dict) or validation.get('checkpoint')!=selected_model:
            validation=None
        return {'csrf': self.csrf, **monitor_status(), 'robot_busy': robot_busy(),
                'phase': phase, 'active': active_run, 'owned': owned,
                'episode': view, 'worker': current or last_worker or None, 'checkpoint': selected_model,
                'checkpoint_contract':selected_checkpoint_contract(selected_model,view),
                'checkpoint_validation':validation,
                'replays': dict(self.replays), 'reviews': dict(self.reviews), 'default_task': DEFAULT_TASK}

    def start(self, payload):
        task = payload.get('task', DEFAULT_TASK)
        if not isinstance(task, str) or not task.strip() or len(task) > 512:
            raise ValueError('请输入任务指令')
        model = checkpoint()
        if not model:
            raise ValueError('没有找到已配置的训练权重')
        with self.lock:
            if self.current_worker() or active_runtimes() or robot_busy() is not False:
                raise RuntimeError('机械臂已有任务在运行或收尾，请先等待它结束')
            state = monitor_status()
            if not state['camera'].get('ok') or not state['robot'].get('ok'):
                raise RuntimeError('相机或机械臂状态暂不可用')
            name = system.generated('pi05-ui')
            self.last_started = name
            self.launching = subprocess.Popen([PYTHON, '-m', 'fr5_runtime.policy_console', '--worker', name,
                                               '--checkpoint', model, '--task', task.strip()],
                                              cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL, start_new_session=True)
        return {'ok': True, 'episode': name}

    def stop(self):
        with self.lock:
            current = self.current_worker()
            if not current:
                raise RuntimeError('没有由此页面启动的策略任务')
            pid, name = current.get('worker_pid'), current.get('episode')
            if not worker_alive(pid, name):
                raise RuntimeError('该运行进程已经退出')
            os.kill(pid, signal.SIGINT)
        return {'ok': True, 'episode': name}

    def replay(self, payload):
        name = system.simple_name(payload.get('episode'))
        view = episode_view(name)
        if not view or not view.get('summary'):
            raise ValueError('完成保存后才能生成回放')
        with self.lock:
            if self.replays.get(name, {}).get('state') == 'rendering':
                return {'ok': True, 'state': 'rendering'}
            self.replays[name] = {'state': 'rendering'}

        def render():
            try:
                from fr5_runtime import console_data as fr5_console_data
                result = fr5_console_data.render_replay(name)
                item = {'state': 'ready', 'url': 'http://127.0.0.1:8768' + result['video_url']}
            except Exception as exc:
                item = {'state': 'error', 'error': str(exc)}
            with self.lock:
                self.replays[name] = item

        threading.Thread(target=render, daemon=True).start()
        return {'ok': True, 'state': 'rendering'}

    def review(self, payload):
        name = system.simple_name(payload.get('episode'))
        verdict = payload.get('verdict')
        if verdict not in ('success', 'failure'):
            raise ValueError('请选择成功或失败')
        view = episode_view(name)
        if not view or not view.get('summary'):
            raise ValueError('完整运行记录保存后才能评价')
        statement = '用户在策略实验页面判定整轮' + ('成功' if verdict == 'success' else '失败')
        result = system.review(name, verdict, statement, input_method='web_console_click')
        with self.lock:
            self.reviews[name] = verdict
        return {'ok': True, 'verdict': result.get('verdict')}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def trusted_host(self):
        return self.headers.get('Host', '').lower() in (f'127.0.0.1:{self.server.server_port}',
                                                        f'localhost:{self.server.server_port}')

    def respond(self, payload, status=200, kind='application/json; charset=utf-8'):
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if isinstance(payload, dict) else payload
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            if not self.trusted_host():
                return self.respond({'ok': False, 'error': '请使用本机地址'}, 403)
            path = urlparse(self.path).path
            if path == '/':
                return self.respond(PAGE.read_bytes(), kind='text/html; charset=utf-8')
            if path == '/api/status':
                return self.respond(self.server.console.status())
            if path == '/camera.jpg':
                with urllib.request.urlopen(MONITOR + '/color.jpg', timeout=3) as response:
                    return self.respond(response.read(), kind='image/jpeg')
            return self.respond({'ok': False, 'error': '页面不存在'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            return self.respond({'ok': False, 'error': str(exc)}, 503)

    def do_POST(self):
        try:
            expected = 'http://' + self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            token = self.headers.get('X-FR5-CSRF', '')
            if (not self.trusted_host() or (origin and origin != expected)
                    or not secrets.compare_digest(token, self.server.console.csrf)):
                return self.respond({'ok': False, 'error': '请刷新本机页面后操作'}, 403)
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise ValueError('请求格式不正确')
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError('请求格式不正确')
            actions = {'/api/start': self.server.console.start, '/api/stop': lambda _: self.server.console.stop(),
                       '/api/replay': self.server.console.replay, '/api/review': self.server.console.review}
            action = actions.get(urlparse(self.path).path)
            if action is None:
                return self.respond({'ok': False, 'error': '操作不存在'}, 404)
            return self.respond(action(payload))
        except (ValueError, TypeError, KeyError) as exc:
            return self.respond({'ok': False, 'error': str(exc)}, 400)
        except Exception as exc:
            return self.respond({'ok': False, 'error': str(exc)}, 409)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--worker')
    parser.add_argument('--checkpoint')
    parser.add_argument('--task')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.checkpoint, args.task)
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.console = PolicyConsole()
    print(f'策略实验工作台：http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
