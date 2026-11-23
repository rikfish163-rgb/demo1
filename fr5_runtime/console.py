"""Local collection console. GET requests never issue robot commands.

Only an explicit start click starts one complete demonstration. Cup placement
is manual; the click means the user has finished placement and left the area.
Native faults/stop and the existing command recorder remain authoritative.
"""
from __future__ import annotations

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import signal
import threading
import time
import urllib.request
from urllib.parse import urlparse

from fr5_runtime import system
from fr5_runtime import console_data as data

PAGE = system.ROOT/'fr5_lerobot/src/lerobot_robot_fairino/web/collection.html'
STATE_DIR = system.LOCAL_STATE/'console'
REFERENCE = system.SPOOL/'reference-success-000001'
BUSY = {'preparing', 'running', 'saving', 'stopping'}
PHASES = {'initial_hold':'开始记录', 'initial_open':'张开夹爪',
          'approach_clear':'接近杯子', 'approach_near':'接近杯子',
          'approach_contact':'对准杯身', 'close':'夹取',
          'lift_probe':'抬起杯子', 'lift_full':'抬起并悬停',
          'return':'放回桌面', 'release':'松开夹爪',
          'retreat_probe':'退开', 'retreat_clear':'退到杯旁', 'retreat_high':'退开'}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temporary.replace(path)


def closed_stream_receipt(root):
    """Read a complete, server-acknowledged close receipt without opening a stream."""
    root = Path(root)
    path = root/'stream-closed.json'
    if path.is_symlink() or not path.is_file():
        raise ValueError('缺少原始流关闭记录')
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError('原始流关闭记录无效')
    frames = receipt.get('accepted_frames')
    if (receipt.get('schema') != 'fr5-stream-closed-v1'
            or type(frames) is not int or frames < 1
            or type(receipt.get('last_durable_sequence')) is not int
            or receipt['last_durable_sequence'] != frames-1
            or type(receipt.get('unacknowledged_frames')) is not int
            or receipt['unacknowledged_frames'] != 0
            or type(receipt.get('unacknowledged_bytes')) is not int
            or receipt['unacknowledged_bytes'] != 0
            or 'error' not in receipt or receipt['error'] is not None
            or not isinstance(receipt.get('session'), str)
            or not re.fullmatch(r'[0-9a-f]{32}', receipt['session'])
            or any(not isinstance(receipt.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', receipt[key])
                   for key in ('codec_sha256', 'protocol_sha256'))
            or receipt.get('local_episode') != str(root)):
        raise ValueError('原始流未完整确认，不能作为已完成记录')
    return receipt


def closed_episode_root(name):
    name = system.simple_name(name)
    base = system.SPOOL
    root = base/name
    if (base.is_symlink() or not base.is_dir() or root.is_symlink() or not root.is_dir()
            or root.resolve().parent != base.resolve()):
        raise ValueError('只能恢复本机采集目录中的记录')
    closed_stream_receipt(root)
    return root


def commands_match_scope(actions, journal, count):
    """Allow planned no-op targets and the optional initial empty-gripper opening."""
    if (not isinstance(actions, list) or not actions or not isinstance(journal, list)
            or any(not isinstance(item, dict) for item in actions + journal)):
        return False
    executed = [item for item in journal if item.get('executed') is True]
    if type(count) is not int or count < 1 or len(executed) != count:
        return False
    planned = [(item.get('cycle'), item.get('phase'), item.get('kind')) for item in actions]
    if any(type(cycle) is not int or cycle < 1 or not isinstance(phase, str) or not phase
           or kind not in ('arm', 'gripper') for cycle, phase, kind in planned):
        return False
    actual = []
    for item in executed:
        teacher = item.get('teacher')
        if not isinstance(teacher, dict):
            return False
        actual.append((teacher.get('cycle'), teacher.get('phase'), item.get('kind')))
    if actual and actual[0] == (0, 'initial_open', 'gripper'):
        actual = actual[1:]
    if not actual:
        return False
    remaining = iter(planned)
    return all(any(candidate == command for candidate in remaining) for command in actual)


def completed_summary(root, *, expected_scope=None, expected_scope_sha=None, storage=None):
    """Verify the existing final snapshot; this never labels an episode successful."""
    root = Path(root)
    scope_path, summary_path, journal_path = root/'scope.json', root/'summary.json', root/'target-command-journal.json'
    if any(path.is_symlink() or not path.is_file() for path in (scope_path, summary_path, journal_path)):
        raise ValueError('缺少本轮任务范围或完成摘要')
    scope_raw = scope_path.read_bytes()
    scope, summary = json.loads(scope_raw), json.loads(summary_path.read_text())
    journal = json.loads(journal_path.read_text())
    receipt = closed_stream_receipt(root)
    if not isinstance(scope, dict) or not isinstance(summary, dict):
        raise ValueError('本轮任务范围或完成摘要无效')
    scope_sha = hashlib.sha256(scope_raw).hexdigest()
    actions, stages, task_spec = scope.get('actions'), scope.get('phase_order'), scope.get('task_spec')
    capture, dataset = summary.get('continuous_capture'), summary.get('dataset')
    frames = receipt['accepted_frames']
    if (scope.get('whole_task') is not True or summary.get('whole_task') is not True
            or (expected_scope is not None and scope != expected_scope)
            or (expected_scope_sha is not None and scope_sha != expected_scope_sha)
            or summary.get('scope_sha256') != scope_sha
            or summary.get('outcome') != 'pending_review'
            or 'failed' not in summary or summary['failed'] is not None
            or summary.get('released') is not True
            or not commands_match_scope(actions, journal, summary.get('commands'))
            or not isinstance(stages, list) or not stages or summary.get('completed_stages') != stages
            or not isinstance(task_spec, dict) or summary.get('task_spec') != task_spec
            or type(task_spec.get('repetitions')) is not int
            or type(summary.get('completed_cycles')) is not int
            or summary['completed_cycles'] != task_spec['repetitions']
            or not isinstance(capture, dict) or capture.get('capture_complete') is not True
            or type(capture.get('frames')) is not int or capture['frames'] != frames
            or not isinstance(dataset, dict) or dataset.get('storage_complete') is not True
            or type(dataset.get('recorded_frames')) is not int or dataset['recorded_frames'] != frames):
        raise ValueError('完成摘要与任务范围或原始流不一致')
    if storage is not None:
        spool = getattr(storage, 'frame_spool', None)
        if (spool is None or getattr(spool, 'session', None) != receipt['session']
                or getattr(storage, 'codec_sha256', None) != receipt['codec_sha256']
                or getattr(storage, 'protocol_sha256', None) != receipt['protocol_sha256']):
            raise ValueError('原始流关闭记录与本轮存储不一致')
    return summary


def validate_start(payload):
    uv = payload.get('cup_uv')
    mode = payload.get('target_mode', 'manual' if uv is not None else 'auto')
    hold = payload.get('hold_seconds', 5)
    repetitions = payload.get('repetitions', 1)
    variant = payload.get('start_variant', 'alternating')
    if mode not in ('auto', 'manual'):
        raise ValueError('未知的杯位定位方式')
    if mode == 'manual' and (not isinstance(uv, list) or len(uv) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in uv)
            or not 0 <= uv[0] < 1920 or not 0 <= uv[1] < 1080):
        raise ValueError('请在画面中选择杯口中心')
    if type(hold) not in (int, float) or not math.isfinite(hold) or hold <= 0:
        raise ValueError('悬停时间应为大于零的秒数')
    if type(repetitions) is not int or repetitions not in (1, 2):
        raise ValueError('请选择一轮内抓放一次或两次')
    if variant not in ('alternating', 'left', 'right', 'current'):
        raise ValueError('未知的起点选项')
    return {'cup_uv':[float(x) for x in uv] if mode == 'manual' else None, 'target_mode':mode,
            'hold_seconds':float(hold),
            'repetitions':repetitions, 'start_variant':variant}


def start_waypoints(initial, geometry, variant):
    """Empty-gripper transfer to a cup-relative starting point; no RPC here."""
    import numpy as np
    from fr5_jev.prepare_varied_session import transform
    if variant not in ('left', 'right'):
        raise ValueError('A concrete starting side is required')
    model = geometry['route_model']
    contact = np.asarray(model['contact_pose'], float)
    axis = np.asarray(model['tool_axis_base'], float)
    lateral = transform(contact)[:3, 0]
    target = np.asarray(model['clear_pose'], float).copy()
    target[:3] += -50*axis + (60 if variant == 'left' else -60)*lateral
    target[2] = contact[2] + (90 if variant == 'left' else 110)
    raised = np.asarray(initial, float).copy()
    raised[2] = max(raised[2], target[2], contact[2]+110)
    return [raised.tolist(), target.tolist()]


class Console:
    def __init__(self, *, state_dir=STATE_DIR, refresh=True):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.csrf = secrets.token_urlsafe(32)
        self.job = None
        self.controller = None
        self.session = None
        self.stop_event = threading.Event()
        self.shutdown = threading.Event()
        self.worker = None
        self.inventory = {'storage':{'ok':False}, 'counts':{'success':0,'pending':0,'total':0}, 'episodes':[]}
        self.last_error = None
        self.replay_jobs = set()
        self.archive_jobs = set()
        self.vision_lock = threading.Lock()
        self.vision = {'status':'checking', 'cup_uv':None, 'message':'正在自动定位杯子'}
        if refresh:
            terminal_jobs = []
            for path in self.state_dir.glob('cup-ui-*.json'):
                try:
                    previous = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if (isinstance(previous, dict) and previous.get('kind') == 'collection'
                        and previous.get('id') == path.stem and previous.get('state') in ('completed','failed')
                        and type(previous.get('started_at')) in (int,float)):
                    terminal_jobs.append(previous)
            if terminal_jobs:
                self.job = max(terminal_jobs, key=lambda value:value['started_at'])
            threading.Thread(target=self._refresh_loop, daemon=True).start()
            threading.Thread(target=self._vision_loop, daemon=True).start()

    def _vision_loop(self):
        while not self.shutdown.is_set():
            with self.lock:
                busy = self.job and self.job['state'] in BUSY
            if not busy:
                self.locate()
            self.shutdown.wait(1)

    def locate(self):
        from fr5_jev.cup_localization import snapshot
        with self.vision_lock:
            try:
                value = snapshot()
            except Exception as exc:
                value = {'status':'unavailable','cup_uv':None,'message':'暂时无法自动定位：'+str(exc),
                         'received_at':time.time()}
            with self.lock:
                self.vision = value
        return {'ok':True, 'vision':deepcopy(value)}

    def resolve_target(self, request, observation):
        if request['target_mode'] == 'manual':
            return request['cup_uv'], {'status':'operator_selected','source':'user_cup_selection',
                                       'cup_uv':request['cup_uv']}
        from fr5_jev.cup_localization import from_observation
        value = from_observation(observation)
        write_json(observation/'cup-localization.json', value)
        with self.lock:
            self.vision = value
        if value['status'] != 'found':
            raise RuntimeError(value['message'])
        return value['cup_uv'], value

    def _refresh_loop(self):
        while not self.shutdown.is_set():
            self.refresh()
            self.shutdown.wait(5)

    def refresh(self):
        try:
            value = data.inventory()
            if not value.get('storage', {}).get('ok'):
                raise RuntimeError('暂时无法读取 data4，保留上次记录')
            with self.lock:
                self.inventory = value
                self.last_error = None
        except Exception as exc:
            with self.lock:
                self.last_error = '数据目录暂时不可用：'+str(exc)
                self.inventory['storage']['ok'] = False

    def status(self):
        try:
            monitor = json.load(urllib.request.urlopen('http://127.0.0.1:8765/status', timeout=.8))
        except Exception:
            monitor = {'camera':{'ok':False}, 'robot':{'ok':False}}
        with self.lock:
            result = deepcopy(self.inventory)
            job = deepcopy(self.job)
            if job:
                job['elapsed_s'] = (job.get('finished_at') or time.time())-job['started_at']
            result.update(csrf=self.csrf, camera=monitor.get('camera', {'ok':False}),
                          robot=monitor.get('robot', {'ok':False}), job=job, last_error=self.last_error,
                          vision=deepcopy(self.vision))
            for episode in result.get('episodes', []):
                if not isinstance(episode, dict):
                    continue
                name = episode.get('id')
                available = (isinstance(name, str) and episode.get('can_review') is not True
                             and name not in self.archive_jobs
                             and not (self.job and self.job.get('id') == name and self.job.get('state') in BUSY))
                if available:
                    try:
                        closed_episode_root(name)
                    except (OSError, ValueError, TypeError):
                        available = False
                episode['can_retry_save'] = available
        return result

    def update(self, **fields):
        with self.lock:
            if self.job is not None:
                self.job.update(fields)
                write_json(self.state_dir/(self.job['id']+'.json'), self.job)

    def start(self, payload):
        request = validate_start(payload)
        with self.lock:
            if self.job and self.job['state'] in BUSY:
                raise RuntimeError('本轮仍在执行，请等待退开完成')
            # Starting a second process is rejected by robot_lease as well.
            name = system.generated('cup-ui')
            variant = request['start_variant']
            if variant == 'alternating':
                jobs = 0
                for path in self.state_dir.glob('cup-ui-*.json'):
                    try:
                        previous = json.loads(path.read_text())
                    except (OSError, ValueError):
                        continue
                    if (isinstance(previous, dict) and previous.get('kind') == 'collection'
                            and previous.get('id') == path.stem):
                        jobs += 1
                variant = 'left' if jobs % 2 == 0 else 'right'
            request['resolved_start_variant'] = variant
            self.stop_event = threading.Event()
            self.job = {'id':name, 'kind':'collection', 'state':'preparing', 'phase':'prepare',
                        'label':'准备本轮', 'episode':name, 'started_at':time.time(),
                        'completed_actions':0, 'total_actions':0, 'message':'读取当前杯位与设备状态',
                        'request':request, 'input_method':'web_console_click',
                        'user_placement_ready':True, 'retreat_mode':'short'}
            write_json(self.state_dir/(name+'.json'), self.job)
            self.worker = threading.Thread(target=self._collect, args=(name, request, self.stop_event), daemon=True)
            self.worker.start()
            return {'ok':True, 'episode':name, 'job_id':name}

    def stop(self):
        with self.lock:
            if not self.job or self.job['state'] not in BUSY:
                return {'ok':True, 'message':'当前没有执行中的动作'}
            self.stop_event.set()
            controller = self.controller
            self.update(state='stopping', label='正在停止', message='保留已采集片段')
        if controller is not None:
            controller.stop('User clicked stop in the collection console')
        return {'ok':True, 'message':'停止请求已送达'}

    def _check_stop(self, stop):
        if stop.is_set():
            raise RuntimeError('用户已停止本轮')

    def _reposition(self, name, plan, variant, stop):
        import numpy as np
        from lerobot_robot_fairino import FairinoFR5, FairinoFR5Config, GeminiBridgeConfig
        from lerobot_robot_fairino.pi05_rollout import PolicyController, DirectSettings, _camera_fresh
        from lerobot_robot_fairino.control_transport import FR5RPC
        from lerobot_robot_fairino.action_control import rpc_vector
        from fr5_jev.whole_task_collection import open_gripper_complete

        class SetupController(PolicyController):
            def _append(self, *args, **kwargs):
                row = super()._append(*args, **kwargs)
                row['source'] = 'llm_workflow_start_setup'
                self._journal[-1]['source'] = row['source']
                return row

        scope = json.loads((plan/'scope.json').read_text())
        geometry = json.loads((plan/'geometry.json').read_text())
        poses = start_waypoints(scope['initial_tcp'], geometry, variant)
        self.update(completed_actions=0, total_actions=len(poses))
        observer = FairinoFR5(FairinoFR5Config(camera=GeminiBridgeConfig(
            url='http://127.0.0.1:8765/snapshot_fast.npz', fps=6)))
        rpc = FR5RPC()
        controller = SetupController(stream=observer.stream, camera_fresh=lambda:_camera_fresh(observer),
            limits=DirectSettings(speed_percent=10.), allow_motion=True, rpc=rpc)
        report = {'source':'current_chat_defined_start_variation', 'variant':variant,
                  'user_input':'web_console_start', 'training_demo':False, 'actions':[]}
        watch_stop = threading.Event()
        watcher = None
        with self.lock:
            self.controller = controller
        try:
            self._check_stop(stop)
            observer.connect(); observer.get_sample()
            controller.arm(operator_confirmed=True)
            self._check_stop(stop)
            state = controller.ready_state()
            report['initial_state'] = state
            if not open_gripper_complete(state):
                if state.get('gripper_active') != 1:
                    raise RuntimeError('夹爪尚未激活，当前开度反馈无效')
                raise RuntimeError('夹爪尚未张开到位，请检查夹爪状态')
            def watch():
                while not watch_stop.wait(.05):
                    controller.watchdog()
                    if stop.is_set():
                        controller.stop('User stopped start-position setup')
                        return
            watcher = threading.Thread(target=watch, daemon=True); watcher.start()
            for pose in poses:
                self._check_stop(stop)
                state = controller.ready_state()
                q = rpc_vector(rpc.call('GetInverseKinRef', 0, pose, state['joints']), 6, 'start-position IK')
                target = [*q.tolist(), state['gripper_position']]
                row = {'tcp_target':pose, 'target':target, 'status':'prepared'}
                report['actions'].append(row); write_json(plan/'start-setup.json', report)
                row['result'] = controller.execute(target, kind='arm')
                row['status'] = row['result']['status']
                write_json(plan/'start-setup.json', report)
                self.update(completed_actions=sum(x['status'] == 'completed' for x in report['actions']))
            report.update(status='completed', final_state=controller.ready_state(), journal=deepcopy(list(controller._journal)))
        except BaseException as exc:
            report.update(status='interrupted', error=str(exc), journal=deepcopy(list(controller._journal)))
            controller.stop(str(exc))
            raise
        finally:
            watch_stop.set()
            if watcher:
                watcher.join(2)
            try:
                write_json(plan/'start-setup.json', report)
            finally:
                # A failed report write must not keep the robot state stream open.
                try:
                    if observer.is_connected:
                        observer.disconnect()
                finally:
                    with self.lock:
                        if self.controller is controller:
                            self.controller = None

    def _collect(self, name, request, stop):
        from fr5_jev.whole_task_collection import WholeTaskSession, prepare, record_llm_route_plan
        console = self

        class ConsoleSession(WholeTaskSession):
            defer_materialization = True

            def select_phase(self, phase, cycle, **kwargs):
                console.update(state='running', phase=phase, label=PHASES.get(phase, phase), message='正在连续采集')
                return super().select_phase(phase, cycle, **kwargs)

            def dispatch(self, action):
                result = super().dispatch(action)
                count = sum(x.get('executed') is True for x in self.journal)
                console.update(completed_actions=count)
                return result

            def hold(self):
                console.update(phase='hold', label='空中悬停', message=f"按任务悬停 {self.scope['hold_seconds']:g} 秒")
                return super().hold()

            def finish(self, *args, **kwargs):
                console.update(state='saving', label='保存本轮', message='确认原始记录已写入 data4')
                try:
                    return super().finish(*args, **kwargs)
                except Exception as exc:
                    outcome = args[0] if args else kwargs.get('outcome', 'interrupted')
                    if outcome != 'pending_review' or self.failed is not None:
                        raise
                    try:
                        summary = completed_summary(self.root, expected_scope=self.scope,
                                                    expected_scope_sha=self.scope_sha, storage=self.storage)
                    except (OSError, ValueError, TypeError, KeyError):
                        raise exc
                    self._console_storage_error = str(exc)
                    return summary

            def final_summary_fields(self):
                return {**super().final_summary_fields(), 'materialization_deferred':True,
                        'collection_interface':'web_console', 'start_variant':request['resolved_start_variant']}

        session = None
        observations = []
        plan_root = None
        try:
            with system.robot_lease():
                self._check_stop(stop)
                observation = Path(system.observe()['directory']); observations.append(observation)
                self.update(label='定位杯子', phase='localize', message='使用本轮最新画面定位杯口')
                cup_uv, localization = self.resolve_target(request, observation)
                if request['resolved_start_variant'] != 'current':
                    plan_root = system.SPOOL/('setup-'+name)
                    prepare(plan_root, observation, cup_uv, grasp_reference=REFERENCE,
                            hold_seconds=request['hold_seconds'], repetitions=request['repetitions'], retreat_mode='short')
                    self._check_stop(stop)
                    self.update(label='变换机械臂起点', phase='start_setup', message='杯子保持原位，空夹爪转移到新起点')
                    self._reposition(name, plan_root, request['resolved_start_variant'], stop)
                    observation = Path(system.observe()['directory']); observations.append(observation)
                    cup_uv, localization = self.resolve_target(request, observation)
                self._check_stop(stop)
                root = system.SPOOL/name
                prepare(root, observation, cup_uv, grasp_reference=REFERENCE,
                        hold_seconds=request['hold_seconds'], repetitions=request['repetitions'], retreat_mode='short')
                automatic = request['target_mode'] == 'auto'
                plan_source = ('reused_llm_workflow_with_visual_localization' if automatic
                               else 'reused_llm_workflow_with_user_cup_selection')
                target_note = ('杯口由本轮全画面 RGB-D 外缘与已知尺寸自动定位；用户只点击摆好开始。'
                               if automatic else '用户在网页选择本轮杯口并点击摆好开始。')
                record_llm_route_plan(root,
                    '复用当前对话 LLM 已定义的杯外接近、抓起、悬停、原位放回和短退开流程。'
                    +target_note+'本轮坐标由当前 RGB-D 与实际机械臂状态解算。'
                    '没有新的视觉 LLM API 调用。空间条件依据用户在旁监督，不声明独立全路径碰撞验证。',
                    source=plan_source)
                write_json(root/'console-input.json', {'request':request, 'input_method':'web_console_click',
                           'ready_clicked_unix':self.job['started_at'], 'plan_source':plan_source,
                           'resolved_cup_uv':cup_uv, 'localization':localization})
                write_json(root/'cup-localization.json', localization)
                session = ConsoleSession(root)
                session.policy_name = ('reused_llm_workflow+jev+automatic_visual_localization' if automatic
                                       else 'reused_llm_workflow+jev+user_cup_selection')
                session.interrupted = stop
                with self.lock:
                    self.session = session
                self.update(completed_actions=0, total_actions=len(session.scope['actions']))
                try:
                    self._check_stop(stop)
                    session.start()
                    with self.lock:
                        self.controller = session.controller
                    self._check_stop(stop)
                    summary = session.run()
                    if summary.get('outcome') == 'pending_review':
                        if summary != completed_summary(session.root, expected_scope=session.scope,
                                                        expected_scope_sha=session.scope_sha, storage=session.storage):
                            raise RuntimeError('返回的完成摘要与本地记录不一致')
                except BaseException as exc:
                    session.failed = str(exc)
                    if session.controller is not None:
                        session.controller.stop(str(exc))
                    if session.writer is None or not session.writer._finished:
                        session.finish('interrupted')
                    raise
                finally:
                    session._watchdog_stop.set()
                    try:
                        if session.robot.is_connected:
                            session.robot.disconnect()
                    finally:
                        with self.lock:
                            self.controller = None
                            self.session = None
                if summary['outcome'] != 'pending_review':
                    raise RuntimeError(summary.get('failed') or '本轮中断，未计为成功')
            # The robot lease is released before media reconstruction/encoding.
            self.update(state='completed', phase='completed', storage_status='pending',
                        label='动作完成，记录待保存',
                        message='原始流已确认，记录正在后台保存；保存后仍需用户整轮确认', finished_at=time.time())
        except BaseException as exc:
            self.update(state='failed', label='本轮已停止', message=str(exc), finished_at=time.time())
        finally:
            if session is not None and session.storage is not None:
                try:
                    closed_stream_receipt(session.root)
                except (OSError, ValueError, TypeError):
                    pass
                else:
                    self._schedule_archive(name, lambda: session.storage, observations, plan_root)

    def _storage_update(self, name, status):
        """Update this episode's job, without replacing a later active job."""
        with self.lock:
            current = self.job is not None and self.job.get('id') == name
            path = self.state_dir/(name+'.json')
            if current:
                job = self.job
            else:
                try:
                    job = json.loads(path.read_text())
                except (OSError, ValueError):
                    return
            if not isinstance(job, dict) or job.get('id') != name:
                return
            job['storage_status'] = status
            if job.get('state') == 'completed':
                if status == 'saved':
                    job.update(label='动作完成，待整轮确认', message='记录已保存，请由用户确认整轮结果')
                else:
                    job.update(label='动作完成，记录待保存',
                               message=('记录正在后台保存，保存后仍需用户整轮确认' if status == 'pending'
                                        else '记录保存未完成，请重试保存；尚不能确认成功'))
            write_json(path, job)

    def _archive_failed(self, name, exc):
        write_json(self.state_dir/(name+'-archive.json'), {'status':'failed', 'error':str(exc),
                   'raw_stream_retained':True})
        self._storage_update(name, 'failed')

    def _schedule_archive(self, name, storage_factory, observations, plan_root):
        with self.lock:
            if name in self.archive_jobs:
                return False
            self.archive_jobs.add(name)
            try:
                self._storage_update(name, 'pending')
            except BaseException:
                self.archive_jobs.discard(name)
                raise
        def run():
            try:
                self._archive(name, storage_factory(), observations, plan_root)
            except Exception as exc:
                self._archive_failed(name, exc)
            finally:
                with self.lock:
                    self.archive_jobs.discard(name)
        try:
            threading.Thread(target=run, daemon=True).start()
        except BaseException as exc:
            with self.lock:
                self.archive_jobs.discard(name)
            if isinstance(exc, Exception):
                self._archive_failed(name, exc)
            raise
        return True

    def _archive(self, name, storage, observations, plan_root):
        try:
            storage.finalize_control()
            receipt = storage.materialize()
            write_json(self.state_dir/(name+'-archive.json'), {'status':'materialized', 'receipt':receipt})
            self._storage_update(name, 'saved')
        except Exception as exc:
            self._archive_failed(name, exc)
            return
        try:
            storage.release_checkpoint_caches(receipt)
            self.refresh()
            if observations or plan_root:
                self._archive_auxiliary(name, observations, plan_root)
            data.render_replay(name)
            self.refresh()
        except Exception as exc:
            write_json(self.state_dir/(name+'-auxiliary.json'), {'status':'local_cache_retained','error':str(exc)})

    def retry_save(self, payload):
        name = system.simple_name(payload.get('episode'))
        with self.lock:
            if self.job and self.job.get('id') == name and self.job.get('state') in BUSY:
                raise RuntimeError('本轮仍在收尾，请等待后再重试保存')
        root = closed_episode_root(name)
        try:
            previous = json.loads((self.state_dir/(name+'-archive.json')).read_text())
        except (OSError, ValueError):
            previous = {}
        if isinstance(previous, dict) and previous.get('status') == 'materialized':
            self._storage_update(name, 'saved')
            return {'ok':True, 'episode':name, 'storage_status':'saved', 'message':'记录已保存，等待用户整轮确认'}
        from fr5_jev.stream_storage import StreamStorage
        scheduled = self._schedule_archive(name, lambda: StreamStorage.from_closed(root), [], None)
        return {'ok':True, 'episode':name, 'storage_status':'pending',
                'message':'正在重新保存记录' if scheduled else '该记录正在保存'}

    def _archive_auxiliary(self, name, observations, plan_root):
        import hashlib
        import io
        import shlex
        import subprocess
        import tarfile
        files = {}
        for index, directory in enumerate([*observations, *([plan_root] if plan_root else [])]):
            if not directory.resolve().is_relative_to(system.SPOOL.resolve()):
                raise ValueError('辅助记录目录不属于本轮缓存')
            for path in directory.iterdir():
                if path.is_file() and not path.is_symlink():
                    files[f'{index:02d}/{path.name}'] = path
        metadata = {key:{'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                    for key,p in files.items()}
        memory = io.BytesIO()
        with tarfile.open(fileobj=memory, mode='w:gz') as archive:
            for key,path in files.items():
                archive.add(path, arcname=key)
        destination = system.REMOTE+'/observations/'+system.simple_name(name)
        subprocess.run(system.ssh_args('mkdir -p '+shlex.quote(system.REMOTE+'/observations')+
            ' && mkdir '+shlex.quote(destination)+' && tar -xz -C '+shlex.quote(destination)),
            input=memory.getvalue(), check=True, capture_output=True, timeout=90)
        proof = system.remote_json(r'''
import json,sys,hashlib
from pathlib import Path
q=json.load(sys.stdin);p=Path(q['destination'])
for name,item in q['files'].items():
 f=p/name
 if f.stat().st_size!=item['bytes'] or hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Auxiliary copy differs')
(p/'archive-manifest.json').write_text(json.dumps(q['files'],indent=2))
print(json.dumps({'status':'verified','files':len(q['files']),'destination':str(p)}))
''', {'destination':destination,'files':metadata})
        for path in files.values():
            if path.name == 'scene.npz':
                path.unlink()
        write_json(self.state_dir/(name+'-auxiliary.json'), proof)

    def review(self, payload):
        episode = system.simple_name(payload.get('episode'))
        verdict = payload.get('verdict')
        statement = payload.get('statement')
        if verdict not in ('success', 'failure') or not isinstance(statement, str) or not statement.strip():
            raise ValueError('请选择整轮成功或失败')
        if verdict == 'success':
            with self.lock:
                path = self.state_dir/(episode+'.json')
                if self.job and self.job.get('id') == episode:
                    job = self.job
                else:
                    try:
                        job = json.loads(path.read_text())
                    except (OSError, ValueError):
                        job = {}
                status = job.get('storage_status') if isinstance(job, dict) and job.get('id') == episode else None
                try:
                    archive = json.loads((self.state_dir/(episode+'-archive.json')).read_text())
                except (OSError, ValueError):
                    archive = {}
                verified = (status == 'saved' or (isinstance(archive, dict)
                            and archive.get('status') == 'materialized')
                            or any(item.get('id') == episode and item.get('can_review') is True
                                   for item in self.inventory.get('episodes', []) if isinstance(item, dict)))
                local_closed = (system.SPOOL/episode/'stream-closed.json').is_file()
                if not verified and (episode in self.archive_jobs or status in ('pending', 'failed') or local_closed):
                    raise RuntimeError('记录尚未保存完成，不能确认成功')
        result = system.review(episode, verdict, statement, input_method='web_console_click')
        with self.lock:
            if self.job and self.job.get('id') == episode and self.job.get('state') == 'completed':
                self.update(label='整轮已确认成功' if verdict == 'success' else '整轮已确认失败',
                            message='用户评价已保存')
        self.refresh()
        return {'ok':True, 'review':result}

    def replay(self, payload):
        name = system.simple_name(payload.get('episode'))
        with self.lock:
            if name in self.replay_jobs:
                return {'ok':True, 'message':'正在整理回放'}
            self.replay_jobs.add(name)
        def render():
            try:
                data.render_replay(name); self.refresh()
            except Exception as exc:
                with self.lock:
                    self.last_error = '生成回放失败：'+str(exc)
            finally:
                with self.lock:
                    self.replay_jobs.discard(name)
        threading.Thread(target=render, daemon=True).start()
        return {'ok':True, 'message':'正在整理回放'}


def byte_range(header, size):
    if not header:
        return 0, size-1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not any(match.groups()):
        raise ValueError('Invalid range')
    a, b = match.groups()
    if not a:
        start, end = max(0, size-int(b)), size-1
    else:
        start, end = int(a), min(size-1, int(b) if b else size-1)
    if not 0 <= start <= end < size:
        raise ValueError('Invalid range')
    return start, end, True


class Handler(BaseHTTPRequestHandler):
    server_version = 'FR5Console/1.0'

    def log_message(self, *args):
        pass

    def trusted_host(self):
        return self.headers.get('Host', '').lower() in (
            f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def respond(self, value, status=200):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers(); self.wfile.write(content)

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        path = urlparse(self.path).path
        try:
            if not self.trusted_host():
                self.respond({'ok':False, 'error':'请使用本机控制台地址'}, 403); return
            if path == '/':
                raw = PAGE.read_bytes(); kind = 'text/html; charset=utf-8'
            elif path == '/api/status':
                self.respond(self.server.console.status()); return
            elif path == '/camera.jpg':
                raw = urllib.request.urlopen('http://127.0.0.1:8765/color.jpg', timeout=2).read()
                kind = 'image/jpeg'
            elif path.startswith('/media/') and path.endswith('.mp4'):
                self.media(path, head); return
            else:
                self.respond({'ok':False, 'error':'页面不存在'}, 404); return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            if not head:
                self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.respond({'ok':False, 'error':str(exc)}, 503)

    def media(self, path, head):
        name = system.simple_name(path[len('/media/'):-len('.mp4')])
        meta = data.media_metadata(name)
        size = meta['size']
        try:
            start, end, partial = byte_range(self.headers.get('Range'), size)
        except ValueError:
            self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers(); return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end-start+1))
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head:
            return
        process = data.open_media(name, start, end)
        try:
            while True:
                chunk = process.stdout.read(128*1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)

    def do_POST(self):
        try:
            origin = self.headers.get('Origin')
            expected = f'http://{self.headers.get("Host")}'
            if (not self.trusted_host() or (origin and origin != expected)
                    or not secrets.compare_digest(self.headers.get('X-FR5-CSRF', ''), self.server.console.csrf)):
                self.respond({'ok':False, 'error':'请刷新本机控制台后操作'}, 403); return
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise ValueError('请求格式不正确')
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError('请求格式不正确')
            actions = {'/api/start':self.server.console.start, '/api/stop':lambda _:self.server.console.stop(),
                       '/api/locate':lambda _:self.server.console.locate(),
                       '/api/review':self.server.console.review, '/api/replay':self.server.console.replay,
                       '/api/retry_save':self.server.console.retry_save}
            operation = actions.get(urlparse(self.path).path)
            if operation is None:
                self.respond({'ok':False, 'error':'操作不存在'}, 404); return
            self.respond(operation(payload))
        except (ValueError, TypeError, KeyError) as exc:
            self.respond({'ok':False, 'error':str(exc)}, 400)
        except Exception as exc:
            self.respond({'ok':False, 'error':str(exc)}, 409)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.console = Console()
    def stop(*_):
        server.console.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    print(f'Collection console http://127.0.0.1:{args.port}/ (no motion on startup)', flush=True)
    try:
        server.serve_forever()
    finally:
        server.console.shutdown.set()
        server.console.stop()
        if server.console.worker and server.console.worker.is_alive():
            server.console.worker.join(30)
        server.server_close()


if __name__ == '__main__':
    main()
