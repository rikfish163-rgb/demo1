"""Standalone pi0.5: ordered chunks, fixed-rate targets, native FR5 servo.

Commands come from a complete π0.5 draw. An operator stop or an explicitly
supplied duration ends a run; neither verifies task completion.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import signal
import threading
import time

from .stream_contract import CONTRACT, FPS, CHUNK, CUP_RELATIVE_WIRE, ABSOLUTE_JOINT_WIRE, decode_action
from .stream_policy import StreamPolicy, make_request, validate_reply
from .cup_action_selection import choose_proposal
from fr5_lerobot.training.stream_observation_history import (
    CUP_CAMERA_CLOCK_PROFILES, CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE, CUP_SETDOWN_PROFILES,
    CUP_HISTORY_PROFILES, CUP_TCP_HISTORY_PROFILES,
    CUP_TIMING_PROFILES, CUP_TRACKING_PROFILES, HISTORY_PROFILES,
)


MAX_TARGET_LATENESS_S = 3.0
MAX_JOINT_STEP_DEG = 1.5
# The ten reviewed grasps command 76%, with contact reported at 78–79%.
# The effective target can stay at 77% under the 1 percentage point deadband.
CUP_FINAL_CLOSE_PERCENT = 77
_CUP_BOUNDARY_WEIGHTS = (
    '769b3f8efc2b6a977e112ed1ff9bdfd0bac51cb7c736a922197db3e8a363d43d',
    'e6970b38ede7d8cd75dc108317f4bc4064e6f9e6c111e5207236c730e6e21317',
)


class CupProposalRejected(RuntimeError):
    """Keep rejected model draws available to the evaluation recorder."""

    def __init__(self, reply):
        super().__init__('π0.5 did not propose a feasible cup boundary action in three draws')
        self.reply = reply


def predict_stream_reply(policy, request, initial_cup):
    profile = request.get('observation_context_profile')
    if (profile not in (CUP_SETDOWN_PROFILE, CUP_SETDOWN_FIXED_PROFILE) or
            policy.n_action_steps != 3 or
            (profile == CUP_SETDOWN_PROFILE and policy.checkpoint_sha not in _CUP_BOUNDARY_WEIGHTS)):
        return policy.predict(request)
    reply, _, _ = choose_proposal(policy, request, initial_cup['table_normal'], check_boundaries=True)
    if not reply.get('cup_proposal_selection', {}).get('matched', True):
        raise CupProposalRejected(reply)
    return reply


def require_cup_contact(completion, target_percent, submitted_at):
    if (not isinstance(completion, dict)
            or completion.get('target_percent') != target_percent
            or completion.get('command_began_monotonic', -math.inf) < submitted_at
            or completion.get('motiondone') != 2):
        raise RuntimeError(f'PGI-140 did not report cup contact at close target {target_percent}%: {completion}')


class OrderedChunk:
    """Consume up to the configured policy prefix in order, starting at zero.

    Skipping a prefix of an unconditioned chunk can skip closing the gripper
    while keeping later actions that assume it closed. Only a validated RTC
    implementation may replace actions that were committed during inference.
    Ordinary joint policies reobserve after an acknowledged gripper command.
    """
    def __init__(self, request, reply, checkpoint_sha, *, execution_steps=CHUNK, initial_cup=None,
                 joint_scene=None):
        self.actions = validate_reply(reply, request, checkpoint_sha)
        if type(execution_steps) is not int or not 1 <= execution_steps <= CHUNK:
            raise ValueError('Checkpoint n_action_steps must be an integer from 1 to 50')
        self.execution_steps = execution_steps
        self.source_time = request['source_time']
        self.request_id = request['id']
        self.anchor_joints = request['state'][:6]
        self.anchor_gripper_fraction = request['state'][6]
        self.gripper_feedback_valid = request['state'][7] == 1
        self.joint_scene = deepcopy(joint_scene)
        self.joint_preflight = None
        self._preflight_request = {'id': request['id'], 'state': list(request['state'])}
        self.cup_actions = None
        self.initial_cup = None
        self.support_reference = None
        self.setdown_release_started = False
        if reply.get('wire_action_representation') == CUP_RELATIVE_WIRE:
            if initial_cup is None or request.get('cup_base_xyz_mm') != initial_cup['base_xyz_mm']:
                raise ValueError('Cup action requires the request and episode to share the measured origin')
            self.initial_cup = initial_cup
            timing = request.get('cup_timing', {})
            if (request.get('cup_tracking', {}).get('native_contact') == 1
                    and timing.get('support_gap_mm') is not None):
                self.support_reference = (
                    request['cup_relative_state'][:3], initial_cup['table_normal'], timing['support_gap_mm'])
        self.cursor = 0

    def prepare(self):
        """Solve the entire consumed prefix before any of it can drive the robot."""
        if self.initial_cup is not None:
            if not self.gripper_feedback_valid:
                raise RuntimeError(f'No settled gripper feedback in policy request {self.request_id}')
            from .control_transport import FR5RPC
            from .cup_relative_runtime import check_cup_close, check_table_clearance, decode_prefix
            rpc = FR5RPC()
            decoded = decode_prefix(self.actions, self.initial_cup, self.anchor_joints, self.execution_steps, rpc=rpc)
            check_cup_close(decoded, gripper_fraction=self.anchor_gripper_fraction,
                            support_reference=self.support_reference)
            clearance = check_table_clearance(self.anchor_joints, decoded, self.initial_cup, rpc=rpc)
            if not clearance['passed']:
                raise RuntimeError(f'Cup-relative prefix table clearance failed: {clearance["error"]}')
            self.cup_actions = decoded
            self.cup_clearance = clearance
        elif self.joint_scene is not None:
            if not self.gripper_feedback_valid:
                raise RuntimeError(f'No settled gripper feedback in policy request {self.request_id}')
            from .control_transport import FR5RPC
            from fr5_lerobot.diagnostics.preflight_jev_pi05_actions import preflight
            self.joint_preflight = preflight(
                self._preflight_request,
                {'id': self.request_id, 'actions': self.actions,
                 'wire_action_representation': ABSOLUTE_JOINT_WIRE},
                self.joint_scene, self.execution_steps, rpc=FR5RPC())
            if not self.joint_preflight['passed']:
                raise RuntimeError(f'Joint prefix preflight failed: {self.joint_preflight["failures"]}')

    def check_release_feedback(self, state):
        from .cup_relative_runtime import check_measured_cup_setdown
        # A partial opening at the table can already clear the contact bit.
        # Subsequent opening ticks still have to stay at the measured endpoint.
        check_measured_cup_setdown(state, self.initial_cup, self.support_reference,
                                   require_contact=not self.setdown_release_started)
        self.setdown_release_started = True

    def _joint_action(self, index):
        if self.initial_cup is not None:
            if self.cup_actions is None:
                raise RuntimeError('Cup-relative prefix has no verified inverse kinematics')
            joint = self.cup_actions[index]['target']
            return [*joint[:6], self.actions[index][6]]
        return self.actions[index]

    @property
    def exhausted(self):
        return self.cursor >= self.execution_steps

    def check_next(self, target_time):
        if self.exhausted:
            raise StopIteration('Action chunk consumed')
        if not self.gripper_feedback_valid:
            raise RuntimeError(f'No settled gripper feedback in policy request {self.request_id}')
        expected = self.source_time + (self.cursor+1)/FPS
        lateness = target_time - expected
        if lateness > MAX_TARGET_LATENESS_S:
            raise RuntimeError(f'Policy target is {lateness:.2f}s late for request {self.request_id}, '
                               f'action {self.cursor} (limit {MAX_TARGET_LATENESS_S:.1f}s)')
        previous = self._joint_action(self.cursor-1)[:6] if self.cursor else self.anchor_joints
        step = max(abs(math.degrees(q-p)) for q, p in zip(self._joint_action(self.cursor)[:6], previous))
        if step > MAX_JOINT_STEP_DEG:
            raise RuntimeError(f'Policy joint step is {step:.2f}° for request {self.request_id}, '
                               f'action {self.cursor} (limit {MAX_JOINT_STEP_DEG:.1f}°)')
        return expected

    def take(self, target_time):
        predicted_time = self.check_next(target_time)
        index = self.cursor
        self.cursor += 1
        raw = self.actions[index]
        joint = self._joint_action(index)
        decoded = decode_action(joint)
        target = [*joint[:6], decoded['gripper_integer_percent']/100.]
        evidence = {'request_id': self.request_id, 'action_index': index,
                        'execution_prefix_steps': self.execution_steps,
                        'prediction_chunk_steps': len(self.actions),
                        'target_monotonic': target_time,
                        'predicted_source_target_monotonic': predicted_time,
                        'raw_action': raw, 'decoded': decoded}
        if self.cup_actions is not None:
            evidence['cup_inverse_kinematics'] = self.cup_actions[index]['detail']
        return target, evidence


def execution_prefix(policy, requested=None):
    """Consume at most the checkpoint's configured prefix before observing."""
    configured = policy.n_action_steps
    if requested is None:
        return configured
    if type(requested) is not int or not 1 <= requested <= configured:
        raise ValueError(f'Execution prefix must be an integer from 1 to {configured}')
    return requested


class InferenceWorker:
    """Inference runs off the stop/control thread and only when requested."""
    def __init__(self, policy, recorder, task, root, *, stop_event=None, joint_scene=None,
                 execution_steps=None):
        self.policy, self.recorder, self.task, self.root = policy, recorder, task, Path(root)
        self.joint_scene = deepcopy(joint_scene)
        self.execution_steps = execution_prefix(policy, execution_steps)
        self.lock = threading.Lock()
        self.stopped = stop_event if stop_event is not None else threading.Event()
        self.requested = threading.Event()
        self.current = self.error = None
        self._next_after_monotonic = None
        self.count = 0
        self.thread = threading.Thread(target=self._run, name='fr5-stream-inference', daemon=True)

    def start(self):
        self.requested.set()
        self.thread.start()

    def request_next(self, after_monotonic):
        if (type(after_monotonic) not in (int, float) or not math.isfinite(after_monotonic)
                or after_monotonic <= 0):
            raise ValueError('A completed target acknowledgement time is required')
        with self.lock:
            if self.stopped.is_set():
                return
            self.current = None
            self._next_after_monotonic = after_monotonic
        self.requested.set()

    def _run(self):
        from fr5_jev.whole_task_collection import durable_json
        from .stream_recording import ObservationCancelled
        try:
            while not self.stopped.is_set():
                if not self.requested.wait(.05):
                    continue
                self.requested.clear()
                if self.stopped.is_set():
                    return
                with self.lock:
                    after_monotonic = self._next_after_monotonic
                    self._next_after_monotonic = None
                started = time.monotonic()
                try:
                    if after_monotonic is None:
                        observation = self.recorder.observation()
                    else:
                        observation = self.recorder.observation(after_monotonic=after_monotonic,
                                                                stop_event=self.stopped)
                except ObservationCancelled:
                    if self.stopped.is_set():
                        return
                    raise
                observation_ready = time.monotonic()
                if self.stopped.is_set():
                    return
                context_profile = getattr(self.policy, 'observation_context_profile', 'none')
                initial_cup = None
                if context_profile in CUP_HISTORY_PROFILES:
                    observation['initial_cup_uv'] = self.recorder.scope['initial_cup']['uv']
                    if context_profile in CUP_TCP_HISTORY_PROFILES:
                        initial_cup = observation['initial_cup'] = self.recorder.scope['initial_cup']
                if context_profile in HISTORY_PROFILES:
                    request = make_request(observation, self.task,
                                           observation_context_profile=context_profile)
                else:
                    request = make_request(observation, self.task)
                if self.stopped.is_set():
                    return
                predict_started = time.monotonic()
                proposal_rejection = None
                try:
                    reply = predict_stream_reply(self.policy, request, initial_cup)
                except CupProposalRejected as exc:
                    reply, proposal_rejection = exc.reply, exc
                chunk = OrderedChunk(request, reply, self.policy.checkpoint_sha,
                                     execution_steps=self.execution_steps, initial_cup=initial_cup,
                                     joint_scene=self.joint_scene)
                received = time.monotonic()
                if self.stopped.is_set():
                    return
                # Save image bytes to server via recording, not duplicated in
                # every small control file. The paired source timestamps link it.
                audit_request = {k: v for k, v in request.items() if k != 'rgb_jpeg_base64'}
                audit = {'request': audit_request, 'reply': reply,
                         'requested_monotonic': started, 'received_monotonic': received,
                         'observation_ready_monotonic': observation_ready,
                         'predict_started_monotonic': predict_started,
                         'source_age_at_reply_s': received-request['source_time'],
                         'execution_prefix_steps': chunk.execution_steps,
                         'status': 'cancelled' if self.stopped.is_set() else 'available',
                         'runtime_llm': False}
                metadata = {'requested_monotonic': started, 'received_monotonic': received,
                            'observation_ready_monotonic': observation_ready,
                            'predict_started_monotonic': predict_started}
                if after_monotonic is not None:
                    audit['observation_after_monotonic'] = after_monotonic
                    metadata['observation_after_monotonic'] = after_monotonic
                self.recorder.record_prediction(request, reply,
                    metadata=metadata)
                if self.stopped.is_set():
                    audit['status'] = 'cancelled'
                try:
                    audit['prefix_precheck_started_monotonic'] = time.monotonic()
                    if proposal_rejection is not None:
                        raise proposal_rejection
                    if not self.stopped.is_set():
                        chunk.prepare()
                    if chunk.cup_actions is not None:
                        audit['cup_table_clearance'] = chunk.cup_clearance
                except Exception as exc:
                    audit.update(status='rejected_before_execution', prefix_precheck_error=str(exc))
                    raise
                finally:
                    audit['prefix_precheck_finished_monotonic'] = time.monotonic()
                    audit['source_age_at_prefix_ready_s'] = (
                        audit['prefix_precheck_finished_monotonic'] - request['source_time'])
                    if chunk.joint_preflight is not None:
                        audit['joint_prefix_preflight'] = chunk.joint_preflight
                    durable_json(self.root/f'chunk-{self.count:05d}.json', audit)
                with self.lock:
                    if self.stopped.is_set():
                        return
                    self.current = chunk
                    self.count += 1
        except BaseException as exc:
            with self.lock:
                self.error = exc

    def latest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f'Stream inference failed: {self.error}') from self.error
            return self.current

    def stop(self):
        self.stopped.set()
        self.requested.set()
        self.recorder.wake_observation_waiters()


def validate_loaded_settings(policy, *, expected_action_steps=None, expected_sha=None):
    """Check what loaded before the selected checkpoint can drive the robot."""
    if expected_sha is not None and policy.checkpoint_sha != expected_sha:
        raise ValueError('Loaded checkpoint SHA differs from the selected bundle')
    if expected_action_steps is not None:
        if type(expected_action_steps) is not int or not 1 <= expected_action_steps <= CHUNK:
            raise ValueError('Invalid selected checkpoint n_action_steps')
        if policy.n_action_steps != expected_action_steps:
            raise ValueError('Loaded service n_action_steps differs from the selected bundle')


def shutdown(policy, robot, recorder, controller, inference, reason, failure):
    """Each cleanup is independent; an uncertain stop is never a clean result."""
    errors, snapshot, summary = [], None, None
    if inference is not None:
        inference.stop()
    if controller is not None:
        try:
            controller.stop(reason or 'runtime cleanup', failure=bool(failure))
        except BaseException as exc:
            errors.append(f'controller stop: {exc}')
        try:
            snapshot = controller.close()
            controller_failure = snapshot.get('failure')
            if controller_failure and (not failure or str(controller_failure) not in failure):
                errors.append(str(controller_failure))
            if (snapshot.get('workers_alive') or snapshot.get('startup_inflight')
                    or snapshot.get('pending_callbacks')
                    or snapshot.get('undelivered_callbacks')
                    or (snapshot.get('last_issued_monotonic') is not None
                        and snapshot.get('stop_result') != 0)):
                errors.append('Controller stop/cleanup was not fully acknowledged: '+str(snapshot))
        except BaseException as exc:
            errors.append(f'controller close: {exc}')
    try:
        policy.close()
    except BaseException as exc:
        errors.append(f'policy close: {exc}')
    worker_alive = False
    if inference is not None:
        inference.thread.join(5.)
        worker_alive = inference.thread.is_alive()
        if worker_alive:
            errors.append('Inference writer still active; evidence was not sealed')
    combined = '; '.join([*([failure] if failure else []), *errors]) or None
    try:
        if recorder is not None:
            if worker_alive:
                # A late writer must never mutate a sealed control manifest.
                # Preserve the incomplete stream for recovery; no new motion.
                recorder.capture._stop.set()
                recorder.capture._thread.join(5.)
                recorder.storage.frame_spool.abort()
            else:
                summary = recorder.finish(reason or 'error', error=combined,
                                          controller_snapshot=snapshot)
    except BaseException as exc:
        errors.append(f'recording close: {exc}')
    finally:
        if robot is not None and robot.is_connected:
            try:
                robot.disconnect()
            except BaseException as exc:
                errors.append(f'observer close: {exc}')
    return summary, errors


def run(checkpoint, server_source, task, directory, *, execute=False, gpu=None,
        seconds=None, loaded_policy=None, expected_action_steps=None, expected_sha=None,
        joint_scene=None, execution_steps=None, **unsupported):
    from .robot_fairino import FairinoFR5
    from .configuration_fairino import FairinoFR5Config
    from .pi05_rollout import _camera_fresh
    from .servo_control import StreamServoController
    from .stream_recording import StreamEvaluationRecording
    from lerobot_camera_gemini import GeminiBridgeConfig
    from fr5_jev.stream_storage import LOCAL_STREAM_BASE
    from fr5_jev.whole_task_collection import durable_json
    if unsupported:
        raise ValueError(f'Unsupported stream runtime options: {list(unsupported)}')
    if seconds is not None and (type(seconds) not in (int, float)
                                or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('An explicit duration must be positive')
    source_dir = Path(__file__).resolve().parent
    source_snapshot = {name: (source_dir/name).read_bytes() for name in
        ('stream_runtime.py','stream_policy.py','stream_contract.py','servo_control.py','stream_recording.py',
         'control_transport.py','pi05_bridge.py','whole_task_policy.py','robot_fairino.py','state_stream.py',
         'continuous_capture.py','feedback.py','cup_action_selection.py')}
    from fr5_lerobot.training import stream_observation_history
    source_snapshot['stream_observation_history.py'] = Path(stream_observation_history.__file__).read_bytes()
    source_snapshot['cup_relative_runtime.py'] = (source_dir/'cup_relative_runtime.py').read_bytes()
    from fr5_lerobot.diagnostics import preflight_jev_pi05_actions
    source_snapshot['preflight_jev_pi05_actions.py'] = Path(preflight_jev_pi05_actions.__file__).read_bytes()
    from fr5_jev import cup_timing, cup_tracking, rgbd_geometry
    for module in (cup_timing, cup_tracking, rgbd_geometry):
        source_path = Path(module.__file__)
        source_snapshot[source_path.name] = source_path.read_bytes()
    root = Path(directory).resolve()
    if root.parent != LOCAL_STREAM_BASE or root.exists():
        raise ValueError('Choose a fresh recording-spool episode directory')
    if loaded_policy is not None and loaded_policy.checkpoint != str(checkpoint):
        raise ValueError('Preloaded model path differs from the requested checkpoint')
    # A diagnostic caller can hand over its already-loaded real server client.
    # This function takes ownership and closes it after execution.
    policy = loaded_policy or StreamPolicy(checkpoint, server_source, gpu=gpu, expected_sha=expected_sha)
    robot = recorder = controller = inference = None
    interrupted = threading.Event()
    stop_watch = threading.Event()
    watcher = None
    old_handlers = {}
    failure, reason, summary = None, None, None
    try:
        while not policy.ready:
            try:
                print(__import__('json').dumps(policy.wait_ready(), ensure_ascii=False), flush=True)
            except TimeoutError:
                print('{"status":"model_loading","robot_commands_sent":0}', flush=True)
        validate_loaded_settings(policy, expected_action_steps=expected_action_steps, expected_sha=expected_sha)
        execution_steps = execution_prefix(policy, execution_steps)
        context_profile = getattr(policy, 'observation_context_profile', 'none')
        if execute and policy.training_data_kind != 'captured':
            raise ValueError('Synthetic diagnostic weights are not captured-demonstration task models')
        if execute and context_profile not in CUP_TCP_HISTORY_PROFILES:
            if joint_scene is None:
                raise ValueError('Joint-policy execution requires a measured table scene')
            preflight_jev_pi05_actions.table_plane(joint_scene)
        robot = FairinoFR5(FairinoFR5Config(camera=GeminiBridgeConfig(
            url='http://127.0.0.1:8765/snapshot_fast.npz', fps=FPS)))
        robot.connect()
        initial_cup = None
        if context_profile in CUP_TCP_HISTORY_PROFILES:
            from .cup_relative_runtime import detect_initial_cup
            initial_cup = detect_initial_cup()
        elif context_profile in CUP_HISTORY_PROFILES:
            from fr5_jev.cup_localization import snapshot
            location = snapshot()
            if location['status'] != 'found':
                raise RuntimeError('An unambiguous initial cup rim observation is required')
            initial_cup = {'uv': location['cup_uv'], 'camera_serial': location['camera_serial'],
                           'observed_monotonic': location['capture_timestamp_monotonic'],
                           'source': location['source']}
        sample = robot.get_sample()
        if execute and context_profile not in CUP_TCP_HISTORY_PROFILES:
            from .action_control import angle_error, finite_vector
            state = sample['robot_state']
            tcp = finite_vector(state['tcp'], 6, 'measured TCP')
            flange = finite_vector(state['flange'], 6, 'measured flange')
            if (state['tool'] != 0 or state['user'] != 0
                    or math.dist(tcp[:3], flange[:3]) > 1.
                    or max(abs(value) for value in angle_error(tcp[3:], flange[3:])) > .1):
                raise ValueError('Joint-policy table preflight requires tool0/user0 flange coordinates')
        if initial_cup is not None:
            from lerobot_camera_gemini.timing import sample_clock
            captured, _, _ = sample_clock(sample)
            if not 0 <= captured - initial_cup['observed_monotonic'] <= 1.0:
                raise RuntimeError('Initial cup and synchronized policy frames are not close in time')
            sample['initial_cup_uv'] = initial_cup['uv']
            sample['initial_cup'] = initial_cup
        root.mkdir(parents=True)
        (root/'runtime-source').mkdir()
        for name, raw in source_snapshot.items():
            (root/'runtime-source'/name).write_bytes(raw)
        if not execute:
            if context_profile in CUP_TRACKING_PROFILES:
                from .cup_relative_runtime import new_cup_tracker, track_cup_sample
                sample['cup_tracking'] = track_cup_sample(new_cup_tracker(initial_cup), sample)
            if context_profile in CUP_TIMING_PROFILES:
                sample['cup_timing'] = cup_timing.CupTiming(initial_cup, aligned=context_profile in CUP_CAMERA_CLOCK_PROFILES,
                    setdown=context_profile in CUP_SETDOWN_PROFILES).update(
                    sample['camera']['metadata'], sample['robot_state'], sample['cup_tracking'])
            request = make_request(sample, task, observation_context_profile=context_profile)
            reply = predict_stream_reply(policy, request, initial_cup)
            result = {'status': 'live_preview_no_motion', 'request': request,
                      'reply': reply, 'robot_commands_sent': 0,
                      'checkpoint_n_action_steps': policy.n_action_steps,
                      'execution_prefix_steps': execution_steps}
            if context_profile in CUP_TCP_HISTORY_PROFILES:
                chunk = OrderedChunk(request, reply, policy.checkpoint_sha, execution_steps=execution_steps,
                                     initial_cup=initial_cup)
                result['initial_cup'] = initial_cup
                try:
                    chunk.prepare()
                except (RuntimeError, ValueError) as exc:
                    result['inverse_kinematics_error'] = str(exc)
                else:
                    result['cup_inverse_kinematics'] = chunk.cup_actions
                    result['cup_table_clearance'] = chunk.cup_clearance
            elif joint_scene is not None:
                result['joint_prefix_preflight'] = preflight_jev_pi05_actions.preflight(
                    request, reply, joint_scene, execution_steps,
                    rpc=preflight_jev_pi05_actions.FR5RPC())
            if context_profile in CUP_TRACKING_PROFILES:
                result['cup_tracking'] = sample['cup_tracking']
            if context_profile in CUP_TIMING_PROFILES:
                result['cup_timing'] = sample['cup_timing']
            durable_json(root/'preview.json', result)
            return result
        scope = {'robot_ip': '192.168.58.2', 'dataset_role': 'policy_evaluation',
                 'policy_contract': deepcopy(CONTRACT), 'checkpoint': str(checkpoint),
                 'checkpoint_sha256': policy.checkpoint_sha, 'task_instruction': task,
                 'runtime_llm': False, 'runtime_jev_calls': 0,
                 'observation_context_profile': context_profile,
                 'wire_action_representation': (CUP_RELATIVE_WIRE if context_profile in CUP_TCP_HISTORY_PROFILES
                                                else ABSOLUTE_JOINT_WIRE),
                 'max_session_s': seconds,
                 'execution_mode': ('ordered_complete_chunk' if execution_steps == CHUNK
                                    else 'ordered_policy_prefix'),
                 'checkpoint_n_action_steps': policy.n_action_steps,
                 'execution_chunk_steps': execution_steps, 'prediction_chunk_steps': CHUNK,
                 'prefix_skip': False,
                 'observation_rule': ('prefix_or_completed_native_gripper_command'
                                      if context_profile not in CUP_HISTORY_PROFILES else 'prefix'),
                 'max_target_lateness_s': MAX_TARGET_LATENESS_S,
                 'max_joint_step_deg': MAX_JOINT_STEP_DEG,
                 'target_delivery': 'ordered_native_acknowledgements',
                 'gripper_deadband_percent': 1,
                 'servo_period_s': .008, 'evidence': [],
                 'runtime_source_sha256': {name:hashlib.sha256(raw).hexdigest()
                                           for name,raw in source_snapshot.items()},
                 'authorization_source': 'existing user authorization for standalone pi05 trials'}
        if initial_cup is not None:
            scope['initial_cup'] = initial_cup
        if joint_scene is not None:
            scope['joint_prefix_preflight_scene'] = deepcopy(joint_scene)
        if context_profile in CUP_TCP_HISTORY_PROFILES:
            scope['cup_final_close_requires_contact_at_or_below_percent'] = CUP_FINAL_CLOSE_PERCENT
        durable_json(root/'scope.json', scope)
        recorder = StreamEvaluationRecording(root, scope)
        recorder.start(robot)
        controller = StreamServoController(robot.stream,
            lambda: _camera_fresh(robot), recorder.command_callback)
        def request_stop(*_):
            interrupted.set()
            controller.stop('operator_stop')
        for sig in (signal.SIGINT, signal.SIGTERM):
            # A Python signal can interrupt the main thread while it holds the
            # driver's RLock. Never perform a reentrant native call from it.
            old_handlers[sig] = signal.signal(sig, lambda *_: interrupted.set())
        def watch_operator():
            while not stop_watch.is_set():
                if interrupted.wait(.005) or (root/'STOP').exists():
                    request_stop()
                    return
        watcher = threading.Thread(target=watch_operator, name='fr5-operator-stop', daemon=True)
        watcher.start()
        inference = InferenceWorker(policy, recorder, task, root, stop_event=interrupted,
                                    joint_scene=joint_scene, execution_steps=execution_steps)
        inference.start()
        # Loading/cold inference is complete before any servo target is sent.
        while inference.latest() is None:
            if interrupted.is_set() or (root/'STOP').exists():
                reason = 'operator_stop'
                break
            recorder.assert_healthy()
            interrupted.wait(.02)
        if interrupted.is_set() or (root/'STOP').exists():
            reason = 'operator_stop'
        if reason is None:
            inference.latest().check_next(time.monotonic()+1/FPS)
            controller.start(operator_confirmed=True)
            began = time.monotonic()
            tick = began
            count = 0
            last_gripper_percent = (controller.snapshot()['gripper_target_percent']
                                    if context_profile in CUP_TCP_HISTORY_PROFILES else None)
            with (root/'submitted-targets.jsonl').open('x') as output, \
                 (root/'acknowledged-targets.jsonl').open('x') as acknowledgements:
                while True:
                    if interrupted.is_set() or (root/'STOP').exists():
                        reason = 'operator_stop'
                        break
                    if seconds is not None and time.monotonic()-began >= seconds:
                        reason = 'explicit_duration_limit'
                        break
                    state = controller.snapshot()
                    if state['status'] != 'running' or state['failure']:
                        raise RuntimeError('Native controller stopped: '+str(state))
                    recorder.assert_healthy()
                    chunk = inference.latest()
                    now = time.monotonic()
                    if chunk is None:
                        # Keep the last commanded position while a fresh chunk
                        # is computed. Operator stop remains checked every 20 ms.
                        interrupted.wait(.02)
                        tick = time.monotonic()
                        continue
                    target, detail = chunk.take(now+1/FPS)
                    release_check = None
                    if last_gripper_percent is not None:
                        from .cup_relative_runtime import check_cup_gripper_change
                        target_percent = detail['decoded']['gripper_integer_percent']
                        check_cup_gripper_change(detail['raw_action'], last_gripper_percent,
                                                 target_percent,
                                                 support_reference=chunk.support_reference)
                        if (chunk.support_reference is not None and last_gripper_percent < 98
                                and target_percent - last_gripper_percent > 1):
                            release_check = chunk.check_release_feedback
                    gripper_commands_before = controller.snapshot()['gripper_commands']
                    token = controller.submit_ordered(target, duration=1/FPS, gripper_precheck=release_check)
                    effective_gripper = controller.snapshot()['gripper_target_percent']
                    detail.update(submitted_monotonic=now, target=target, sequence=count,
                                  delivery_token=token,
                                  prefix_consumed_steps=chunk.cursor,
                                  effective_gripper_target_percent=effective_gripper)
                    check_contact = (last_gripper_percent is not None
                                     and last_gripper_percent > CUP_FINAL_CLOSE_PERCENT >= effective_gripper)
                    output.write(__import__('json').dumps(detail, allow_nan=False)+'\n')
                    output.flush()
                    count += 1
                    if count % FPS == 0:
                        print(__import__('json').dumps({'status': 'policy_stream_progress',
                            'targets_submitted': count, 'chunks_received': inference.count,
                            'elapsed_s': now-began, 'source_age_s': now-chunk.source_time}), flush=True)
                    tick = now+1/FPS
                    # Keep this segment until its native endpoint and changed
                    # gripper command both returned. A slow reply cannot cause
                    # latest-target replacement to erase a close/open command.
                    while True:
                        if interrupted.is_set() or (root/'STOP').exists():
                            request_stop()
                            reason = 'operator_stop'
                            break
                        if seconds is not None and time.monotonic()-began >= seconds:
                            reason = 'explicit_duration_limit'
                            break
                        if controller.target_acknowledged(token) and time.monotonic() >= tick:
                            acknowledged_at = time.monotonic()
                            acknowledged_state = controller.snapshot()
                            gripper_commands_after = acknowledged_state['gripper_commands']
                            if (context_profile not in CUP_HISTORY_PROFILES
                                    and gripper_commands_after > gripper_commands_before):
                                retired_cause = 'native_gripper_command_completed'
                            elif chunk.exhausted:
                                retired_cause = 'execution_prefix_completed'
                            else:
                                retired_cause = None
                            ack = {
                                'sequence':count-1,'delivery_token':token,
                                'acknowledged_monotonic':acknowledged_at,
                                'prefix_consumed_steps':chunk.cursor,
                                'prefix_retired_cause':retired_cause,
                                'gripper_commands_before':gripper_commands_before,
                                'gripper_commands_after':gripper_commands_after,
                                'meaning':'native RPC acknowledgements, not physical task success'}
                            if check_contact:
                                ack['cup_close_feedback'] = acknowledged_state['gripper_completion_feedback']
                            acknowledgements.write(__import__('json').dumps(ack)+'\n')
                            acknowledgements.flush()
                            if check_contact:
                                require_cup_contact(ack['cup_close_feedback'], effective_gripper, now)
                                chunk.setdown_release_started = False
                            if context_profile in CUP_TCP_HISTORY_PROFILES:
                                last_gripper_percent = effective_gripper
                            break
                        interrupted.wait(.008)
                    if reason is not None:
                        break
                    if retired_cause is not None and not interrupted.is_set():
                        inference.request_next(acknowledged_at)
    except BaseException as exc:
        stopped_error = (str(exc) == 'Servo is not running'
                         or str(exc) == 'Controller stopped before ordered target acknowledgement'
                         or str(exc).startswith('Native controller stopped:')
                         or str(exc).startswith('Servo start failed: start cancelled'))
        if (interrupted.is_set() and stopped_error and controller is not None
                and not controller.snapshot().get('failure')):
            reason = 'operator_stop'
        else:
            failure, reason = f'{type(exc).__name__}: {exc}', 'error'
            raise
    finally:
        stop_watch.set()
        if watcher is not None:
            watcher.join(5.)
        try:
            summary, cleanup_errors = shutdown(policy, robot, recorder, controller, inference, reason, failure)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    if cleanup_errors:
        raise RuntimeError('; '.join(cleanup_errors))
    return summary
