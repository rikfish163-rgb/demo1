"""Evaluation evidence for a seven-dimensional streaming policy.

The caller owns the robot, controller and its lease.  This module never opens a
second robot receiver and cannot send a motion command.  Frame payloads use the
existing asynchronous 4090 stream; the protocol's historical ``teacher``
field contains *evaluation observation metadata*, not a teacher demonstration.

After the controller has stopped and drained its callbacks, call ``finish``.
It stops local acquisition, writes small local files and starts a separate
non-daemon thread for network closure.  Release the robot lease immediately;
``wait_finalized`` and ``materialize`` must run outside the lease.
Policy evaluation is never marked as a successful training demonstration.
"""
from __future__ import annotations

import base64
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from lerobot_camera_gemini.timing import MONOTONIC, sample_clock
from .continuous_capture import ContinuousCapture
from .stream_contract import FPS, NAME, contract
from .pi05_bridge import observation_state
from fr5_lerobot.training.stream_observation_history import (
    CUP_CAMERA_CLOCK_PROFILES, CUP_SETDOWN_PROFILES, CUP_TCP_HISTORY_PROFILES, CUP_TIMING_PROFILES,
    CUP_TRACKING_PROFILES, HISTORY_PROFILES, select_observation_history,
)
from fr5_jev.stream_storage import MAX_LOCAL_CONTROL, StreamStorage, new_stream_recording_path

_QUEUE_SIZE = 16384
_QUEUE_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 4 * 1024 * 1024
_END = object()


class ObservationCancelled(InterruptedError):
    """The inference worker stopped while waiting for a recorded observation."""


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def _json(value):
    return json.dumps(value, default=_json_default, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def _save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as out:
        out.write(_json(value) + '\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def _is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


class _Journal:
    """Small, bounded, nonblocking callback queue with rotated JSONL files."""
    def __init__(self, root):
        self.root = root
        self.queue = queue.Queue(maxsize=_QUEUE_SIZE)
        self.lock = threading.Lock()
        self.counts = {'wire': 0, 'prediction': 0}
        self.written = {'wire': 0, 'prediction': 0}
        self.queued_bytes = 0
        self.error = None
        self.accepting = True
        self.thread = threading.Thread(target=self._run, name='fr5-evaluation-journal', daemon=True)
        for name in ('wire', 'prediction'):
            (root / ('stream-' + name)).mkdir(exist_ok=False)
        self.thread.start()

    def add(self, kind, value):
        # Assign callback order while holding the same lock as queue insertion.
        # The real issued timestamps may arrive out of order across channels.
        with self.lock:
            if not self.accepting or self.error:
                raise RuntimeError('Evaluation journal unavailable: ' + str(self.error or 'closed'))
            value = dict(value, callback_sequence=self.counts[kind],
                         callback_monotonic=time.monotonic())
            line = (_json(value) + '\n').encode('utf-8')
            if len(line) > _CHUNK_BYTES or self.queued_bytes + len(line) > _QUEUE_BYTES:
                self.error = 'Evaluation journal exceeded its bounded local queue capacity'
                raise RuntimeError(self.error)
            try:
                self.queue.put_nowait((kind, line))
            except queue.Full as exc:
                self.error = 'Evaluation journal queue full; no record was silently discarded'
                raise RuntimeError(self.error) from exc
            self.queued_bytes += len(line)
            self.counts[kind] += 1

    def _check_local_capacity(self):
        # The existing StreamStorage limit applies to all small control files.
        # Check from this worker (never the servo sender) before each rotation;
        # reserve space for the current wire/prediction chunks and metadata.
        used = 0
        for path in self.root.rglob('*'):
            rel = path.relative_to(self.root)
            if rel.parts[0] == 'frame-outbox' or rel.as_posix() in (
                    'continuous/capture.jsonl', 'lerobot/meta/capture.jsonl'):
                continue
            if rel.as_posix().startswith('continuous/frames/'):
                continue
            try:
                if path.is_file():
                    used += path.stat().st_size
            except FileNotFoundError:
                continue  # an independent atomic small-file write completed
        if used + 2 * _CHUNK_BYTES > MAX_LOCAL_CONTROL:
            raise OSError('Evaluation control files reached the bounded local storage limit')

    def _run(self):
        files, sizes, indices = {}, {}, {}
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is _END:
                        break
                    kind, line = item
                    size = len(line)
                    if kind not in files or sizes[kind] + size > _CHUNK_BYTES:
                        if kind in files:
                            files[kind].flush()
                            os.fsync(files[kind].fileno())
                            files[kind].close()
                        # Make both currently open lengths visible to stat.
                        for other in files.values():
                            if not other.closed:
                                other.flush()
                        self._check_local_capacity()
                        indices[kind] = indices.get(kind, -1) + 1
                        path = self.root / ('stream-' + kind) / f'{indices[kind]:06d}.jsonl'
                        files[kind] = path.open('xb')
                        sizes[kind] = 0
                    files[kind].write(line)
                    sizes[kind] += size
                    self.written[kind] += 1
                finally:
                    if item is not _END:
                        with self.lock:
                            self.queued_bytes -= len(item[1])
                    self.queue.task_done()
        except BaseException as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
        finally:
            for out in files.values():
                if not out.closed:
                    try:
                        out.flush()
                        os.fsync(out.fileno())
                    finally:
                        out.close()

    def close(self):
        with self.lock:
            self.accepting = False
        if self.thread.is_alive():
            try:
                self.queue.put(_END, timeout=5)
            except queue.Full as exc:
                raise RuntimeError('Could not close evaluation journal') from exc
            self.thread.join(10)
        if self.thread.is_alive():
            raise RuntimeError('Evaluation journal worker did not stop')
        if self.error or self.written != self.counts:
            raise RuntimeError(f'Evaluation journal incomplete: {self.error or self.written}')
        for kind in ('wire', 'prediction'):
            directory = self.root / ('stream-' + kind)
            entries = []
            for path in sorted(directory.glob('*.jsonl')):
                raw = path.read_bytes()
                entries.append({'path': path.name, 'bytes': len(raw),
                                'sha256': hashlib.sha256(raw).hexdigest(),
                                'rows': raw.count(b'\n')})
            _save(directory / 'index.json', {'schema': 'fr5-evaluation-' + kind + '-index-v1',
                'count': self.written[kind], 'chunks': entries})


class StreamEvaluationRecording:
    def __init__(self, root, scope):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir() or not isinstance(scope, dict):
            raise ValueError('An existing fresh episode directory and scope are required')
        self.scope = json.loads(_json(scope))
        metadata = self.scope.get('metadata')
        self.read_only_diagnostic = (self.scope.get('model_execution') is False
            or isinstance(metadata, dict) and metadata.get('model_execution') is False)
        self.task = self.scope.get('task_instruction')
        self.checkpoint_sha = self.scope.get('checkpoint_sha256')
        policy = self.scope.get('policy_contract')
        if (not isinstance(self.task, str) or not self.task.strip()
                or not isinstance(self.checkpoint_sha, str) or len(self.checkpoint_sha) != 64
                or any(c not in '0123456789abcdef' for c in self.checkpoint_sha)
                or not isinstance(policy, dict) or policy.get('name') != NAME
                or self.scope.get('dataset_role') != 'policy_evaluation'):
            raise ValueError('Scope requires task, checkpoint SHA, stream contract and policy_evaluation role')
        path = self.root / 'scope.json'
        if path.is_symlink() or not path.is_file() or json.loads(path.read_text()) != self.scope:
            raise ValueError('Saved scope.json must match the supplied scope')
        self.scope_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.capture = self.storage = self.robot = self.journal = None
        self._sample_lock = threading.Lock()
        self._sample_ready = threading.Condition(self._sample_lock)
        self._latest = None
        self.observation_context_profile = scope.get('observation_context_profile', 'none')
        if self.observation_context_profile in CUP_TRACKING_PROFILES:
            from .cup_relative_runtime import new_cup_tracker
            self._cup_tracker = new_cup_tracker(scope['initial_cup'])
        if self.observation_context_profile in CUP_TIMING_PROFILES:
            from fr5_jev.cup_timing import CupTiming
            self._cup_timer = CupTiming(scope['initial_cup'],
                                        aligned=self.observation_context_profile in CUP_CAMERA_CLOCK_PROFILES,
                                        setdown=self.observation_context_profile in CUP_SETDOWN_PROFILES)
        self._state_history = deque(maxlen=128)
        self._frames = 0
        self._finished = False
        self._summary = None
        self._finalized = False
        self._finalize_thread = None
        self._finalize_error = None
        self._closed_receipt = None
        self._finalize_lock = threading.Lock()
        self._method_lock = threading.Lock()
        self._method_counts = {'ServoJ': 0, 'MoveGripper': 0}
        self._start_error = None

    def start(self, robot):
        if self.capture is not None or self._finished:
            raise RuntimeError('Recorder can only start once')
        if not robot.is_connected:
            raise RuntimeError('Connect the shared observer before starting recording')
        self.robot = robot
        # This is a capacity estimate, not a time limit on the task.
        try:
            self.storage = StreamStorage.start_readonly(self.root, self.scope_sha,
                required_bytes=512 * 1024**2, spool_settings={
                    'max_frames': 120, 'max_bytes': 32 * 1024**2, 'max_unacked_age': 30.})
            (self.root / 'SESSION_STARTED').open('x').close()
            self.journal = _Journal(self.root)
            self.capture = ContinuousCapture(robot, self.root / 'continuous', task=self.task,
                fps=FPS, sample_callback=self._sample, frame_spool=self.storage.frame_spool,
                metadata={'scope_sha256': self.scope_sha, 'dataset_role': 'policy_evaluation',
                          'policy_contract': NAME, 'checkpoint_sha256': self.checkpoint_sha,
                          'teacher_protocol_field': 'evaluation frame metadata; not a demonstration'})
            self.capture.start()
            return self
        except BaseException as exc:
            self._start_error = f'{type(exc).__name__}: {exc}'
            if self.journal:
                try:
                    self.journal.close()
                except Exception:
                    pass
            if self.storage is not None:
                self.storage.frame_spool.abort()
            raise

    def _sample(self, sample):
        captured, _, basis = sample_clock(sample)
        if basis != MONOTONIC:
            raise ValueError('Policy evaluation requires camera and robot host monotonic clocks')
        tracking = timing = None
        if self.observation_context_profile in CUP_TRACKING_PROFILES:
            from .cup_relative_runtime import track_cup_sample
            tracking = track_cup_sample(self._cup_tracker, sample)
        if self.observation_context_profile in CUP_TIMING_PROFILES:
            timing = self._cup_timer.update(sample['camera']['metadata'], sample['robot_state'], tracking)
        with self._sample_ready:
            index = self._frames
            self._latest = deepcopy(sample)
            if tracking is not None:
                self._latest['cup_tracking'] = tracking
            if timing is not None:
                self._latest['cup_timing'] = timing
            if self.observation_context_profile in CUP_TCP_HISTORY_PROFILES:
                from .cup_relative_runtime import cup_relative_state
                self._state_history.append((captured, cup_relative_state(sample, self.scope['initial_cup'])))
            elif self.observation_context_profile in HISTORY_PROFILES:
                self._state_history.append((captured, observation_state(sample['robot_state'])))
            self._frames += 1
            self._sample_ready.notify_all()
        row = {'frame_index': index, 'camera': deepcopy(sample['camera']['metadata']),
                'robot': deepcopy(sample['robot_state']), 'sync_clock': basis,
                'sync_skew_s': sample['sync_skew_s'],
                'recording_role': 'policy_evaluation',
                'command_link': 'resolve_actual_wire_events_by_capture_monotonic',
                'captured_monotonic': captured}
        if tracking is not None:
            row['cup_tracking'] = tracking
        if timing is not None:
            row['cup_timing'] = timing
        return row

    def assert_healthy(self):
        if self.capture is None or self._finished:
            raise RuntimeError('Evaluation recorder is not active')
        self.capture.assert_healthy()
        if self.journal.error:
            raise RuntimeError('Evaluation journal failed: ' + self.journal.error)

    def wake_observation_waiters(self):
        """Wake waiters when the inference worker is cancelled."""
        with self._sample_ready:
            self._sample_ready.notify_all()

    def _observation_with_history(self, sample):
        """Copy a sampled observation with causal history under the sample lock."""
        value = deepcopy(sample)
        if value is not None and self.observation_context_profile in HISTORY_PROFILES:
            captured, _, _ = sample_clock(value)
            history = select_observation_history(self._state_history, captured)
            value['observation_history'] = [None if entry is None else
                {'age_s': entry['age_s'], 'state': entry['state']} for entry in history.values()]
        return value

    def observation(self, *, after_monotonic=None, stop_event=None):
        if after_monotonic is not None:
            if not _is_number(after_monotonic) or after_monotonic <= 0:
                raise ValueError('Observation lower bound must be a positive monotonic time')
            # Acquisition uses both existing device timeouts; neither the
            # freshness/skew rules nor gripper-validity rules are changed.
            config = self.robot.config
            deadline = time.monotonic() + config.camera.timeout_s + config.state_timeout_s
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise ObservationCancelled('Observation wait cancelled')
                self.assert_healthy()
                with self._sample_ready:
                    if stop_event is not None and stop_event.is_set():
                        raise ObservationCancelled('Observation wait cancelled')
                    sample = self._latest
                    if sample is not None:
                        captured, robot_time, basis = sample_clock(sample)
                        if basis != MONOTONIC:
                            raise ValueError('Post-ack observation requires host monotonic clocks')
                        if captured > after_monotonic and robot_time > after_monotonic:
                            if not 0 <= time.monotonic() - captured <= .5:
                                raise TimeoutError('Latest recorded observation is stale')
                            return self._observation_with_history(sample)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('No synchronized recorded observation after target acknowledgement')
                    # The condition wakes immediately on a new sample. The
                    # bounded wait also checks device health and cancellation.
                    self._sample_ready.wait(min(remaining, .05))
        self.assert_healthy()
        with self._sample_lock:
            sample = self._observation_with_history(self._latest)
        if sample is None:
            raise RuntimeError('No recorded observation is available')
        captured, _, basis = sample_clock(sample)
        if basis != MONOTONIC or not 0 <= time.monotonic() - captured <= .5:
            raise TimeoutError('Latest recorded observation is stale')
        return sample

    def command_callback(self, row):
        if self.read_only_diagnostic:
            raise PermissionError('Read-only diagnostic does not accept native command records')
        if not isinstance(row, dict):
            raise ValueError('Expected the actual native command record')
        row = json.loads(_json(row))
        method, args = row.get('method'), row.get('args')
        issued, returned = row.get('issued_monotonic'), row.get('returned_monotonic')
        target, code = row.get('target_deg_percent'), row.get('return_code')
        if (method not in ('ServoMoveStart', 'ServoJ', 'MoveGripper', 'StopMotion', 'ServoMoveEnd')
                or not isinstance(args, list) or row.get('raw_args') != args
                or type(row.get('sequence')) is not int or row['sequence'] < 0
                or not _is_number(issued) or issued <= 0 or not _is_number(returned)
                or not _is_number(row.get('issued_unix')) or row['issued_unix'] <= 0
                or not _is_number(row.get('returned_unix')) or row['returned_unix'] <= 0
                or returned < issued or (code is not None and type(code) is not int)
                or (code is None and not row.get('error'))):
            raise ValueError('Invalid native command or result record')
        if method == 'ServoJ':
            if (len(args) != 8 or not isinstance(target, list) or len(target) != 6
                    or any(not _is_number(x) for x in target) or args[0] != target
                    or not isinstance(args[1], list) or len(args[1]) != 4
                    or any(not _is_number(x) for x in args[1] + args[2:])):
                raise ValueError('ServoJ wire arguments do not match issued joints')
        elif method == 'MoveGripper':
            if (len(args) != 10 or type(target) is not int or not 0 <= target <= 100
                    or args[1] != target or any(not _is_number(x) for x in args)):
                raise ValueError('MoveGripper wire arguments do not match issued opening')
        elif args:
            raise ValueError('Native start, end and stop commands take no arguments')
        if self.journal is None:
            raise RuntimeError('Evaluation journal has not started')
        self.journal.add('wire', row)
        if method in self._method_counts:
            with self._method_lock:
                self._method_counts[method] += 1

    def record_prediction(self, request, reply, metadata=None):
        if self.read_only_diagnostic:
            raise PermissionError('Read-only diagnostic does not execute or record a model prediction')
        request, reply = json.loads(_json(request)), json.loads(_json(reply))
        if (not isinstance(request, dict) or not isinstance(reply, dict)
                or request.get('id') != reply.get('id')
                or reply.get('checkpoint_sha256') != self.checkpoint_sha
                or request.get('source_clock') != MONOTONIC):
            raise ValueError('Prediction request, reply or source does not match this evaluation')
        encoded = request.get('rgb_jpeg_base64')
        if not isinstance(encoded, str):
            raise ValueError('Prediction must contain its actual JPEG input')
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('Prediction JPEG base64 is invalid') from exc
        request['rgb_jpeg'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                               'storage': 'exact_base64_in_this_request'}
        self.journal.add('prediction', {'request': request, 'reply': reply,
                                        'execution_metadata': json.loads(_json(metadata or {}))})

    def finish(self, reason, error=None, controller_snapshot=None):
        if self._finished:
            return deepcopy(self._summary)
        if not isinstance(reason, str) or not reason:
            raise ValueError('A termination reason is required')
        if self.read_only_diagnostic and controller_snapshot is not None:
            raise ValueError('Read-only diagnostic must not construct a controller')
        if self._start_error or self.capture is None or self.journal is None:
            # Preserve the initiating failure in the runtime's finally block;
            # no frame stream or hardware command is invented or retried.
            self._summary = {'schema': 'fr5-stream-policy-evaluation-summary-v1',
                'scope_sha256': self.scope_sha, 'dataset_role': 'policy_evaluation',
                'task_instruction': self.task, 'checkpoint_sha256': self.checkpoint_sha,
                'reason': reason, 'error': str(error or self._start_error or 'Recorder not started'),
                'termination_source': reason, 'failed': str(error or self._start_error or 'Recorder not started'),
                'outcome': 'interrupted', 'task_success_verified': False,
                'awaiting_whole_user_review': not self.read_only_diagnostic, 'training_ready': False,
                'read_only_diagnostic': self.read_only_diagnostic,
                'model_execution': False if self.read_only_diagnostic else True,
                'runtime_llm': False, 'recorded_frames': self._frames,
                'wire_records': 0, 'prediction_records': 0, 'wire_callbacks_drained': False,
                'commands': 0, 'summary_stage': 'startup_failed',
                'storage_status': 'startup_failed', 'evidence': []}
            _save(self.root / 'summary.json', self._summary)
            self._finished = True
            return deepcopy(self._summary)
        # ContinuousCapture has no public stop-without-remote-drain operation.
        # Stop its producer now; finish() performs remote acknowledgement later,
        # after the caller has released the hardware lease.
        self.capture._stop.set()
        self.capture._thread.join(2 * self.robot.config.camera.timeout_s + 5)
        if self.capture._thread.is_alive():
            raise RuntimeError('Capture has not stopped; keep the shared observer connected')
        journal_error = None
        try:
            self.journal.close()
        except Exception as exc:
            journal_error = f'{type(exc).__name__}: {exc}'
        snapshot = json.loads(_json(controller_snapshot)) if controller_snapshot is not None else None
        delivered = bool(isinstance(snapshot, dict) and snapshot.get('pending_callbacks') == 0
                         and snapshot.get('inflight_commands', 0) == 0
                         and not snapshot.get('startup_inflight')
                         and not snapshot.get('undelivered_callbacks')
                         and not snapshot.get('callback_failure')
                         and not snapshot.get('workers_alive')
                         and snapshot.get('servo_commands') == self._method_counts['ServoJ']
                         and snapshot.get('gripper_commands') == self._method_counts['MoveGripper']
                         and journal_error is None)
        if self.read_only_diagnostic:
            delivered = None
        self._summary = {'schema': 'fr5-stream-policy-evaluation-summary-v1',
            'scope_sha256': self.scope_sha, 'dataset_role': 'policy_evaluation',
            'task_instruction': self.task, 'checkpoint_sha256': self.checkpoint_sha,
            'reason': reason, 'error': str(error) if error is not None else None,
            'termination_source': reason, 'failed': str(error) if error is not None else journal_error,
            'outcome': ('interrupted' if error is not None or journal_error else
                        'observed' if self.read_only_diagnostic else 'pending_review'),
            'task_success_verified': False, 'awaiting_whole_user_review': not self.read_only_diagnostic,
            'training_ready': False, 'runtime_llm': False,
            'read_only_diagnostic': self.read_only_diagnostic,
            'model_execution': False if self.read_only_diagnostic else True,
            'controller_recording': ('not_applicable_read_only' if self.read_only_diagnostic else 'policy_controller'),
            'recorded_frames': self._frames, 'wire_records': self.journal.counts['wire'],
            'prediction_records': self.journal.counts['prediction'],
            'wire_callbacks_drained': delivered, 'wire_method_counts': self._method_counts.copy(),
            'commands': sum(self._method_counts.values()),
            'summary_stage': 'journal_failed' if journal_error else 'provisional',
            'journal_error': journal_error,
            'controller_snapshot': snapshot,
            'storage_status': 'journal_failed_local_only' if journal_error else 'pending_archive',
            'evidence': []}
        _save(self.root / 'summary.json', self._summary)
        self._finished = True
        if journal_error:
            # Close files and preserve the retained frame outbox without
            # holding the robot lease or claiming a complete archive.
            self._finalize_thread = threading.Thread(target=self._cleanup_failed_journal,
                name='fr5-evaluation-failed-cleanup', daemon=False)
            self._finalize_thread.start()
            raise RuntimeError(journal_error)
        result = deepcopy(self._summary)
        # Threads do not hold the caller's robot lease; being non-daemon keeps
        # the SSH spool alive until durable close if the CLI exits immediately.
        self._finalize_thread = threading.Thread(target=self._background_finalize,
            name='fr5-evaluation-archive', daemon=False)
        self._finalize_thread.start()
        return result

    def _cleanup_failed_journal(self):
        try:
            report = self.capture.finish('interrupted')
            report['dataset_kind'] = 'policy_evaluation_evidence'
            report['training_ready'] = False
            _save(self.root / 'continuous/report.json', report)
        except BaseException:
            pass  # Keep the original journal error and retained local outbox.
        finally:
            self.storage.frame_spool.abort()

    def _background_finalize(self):
        try:
            self.finalize_storage()
        except BaseException as exc:
            self._finalize_error = f'{type(exc).__name__}: {exc}'
            try:
                # Diagnostics must not change a control tree whose manifest
                # may already have been published despite a lost response.
                _save(self.root / 'frame-outbox/archive-error.json', {'error': self._finalize_error,
                      'local_data_preserved': True, 'commands_replayed': 0})
            except OSError:
                pass

    def wait_finalized(self, timeout=None):
        if self._finalize_thread:
            self._finalize_thread.join(timeout)
            if self._finalize_thread.is_alive():
                raise TimeoutError('Background storage closure is still running')
        if self._finalize_error:
            raise RuntimeError('Background storage closure failed: ' + self._finalize_error)
        if not self._finalized:
            raise RuntimeError('Storage closure has not completed')
        return deepcopy(self._summary)

    def _mark_finalized(self):
        """Local completion marker is outside the frozen control manifest."""
        manifest_path = self.root / 'control-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        summary_sha = hashlib.sha256((self.root / 'summary.json').read_bytes()).hexdigest()
        if (manifest.get('scope_sha256') != self.scope_sha
                or manifest.get('summary_sha256') != summary_sha
                or manifest.get('session') != self.storage.frame_spool.session):
            raise RuntimeError('Final manifest does not bind the scope and summary')
        _save(self.root / 'frame-outbox/archive-finalized.json', {
            'schema': 'fr5-evaluation-archive-finalized-v1',
            'session': self.storage.frame_spool.session, 'scope_sha256': self.scope_sha,
            'summary_sha256': summary_sha,
            'control_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'remote_materialization_pending': True})

    def finalize_storage(self):
        """Run after releasing the robot lease; network I/O can take time."""
        with self._finalize_lock:
            if not self._finished:
                raise RuntimeError('Finish local capture before finalizing storage')
            if self._finalized:
                return deepcopy(self._summary)
            if (self.root / 'control-manifest.json').exists():
                # Retry only acknowledgement of the existing immutable
                # snapshot; never rewrite its already-hashed summary.
                self.storage.finalize_control()
                self._summary = json.loads((self.root / 'summary.json').read_text())
                self._mark_finalized()
                self._finalized = True
                self._finalize_error = None
                return deepcopy(self._summary)
            report = self.capture.finish(self._summary['outcome'])
            report['dataset_kind'] = 'policy_evaluation_evidence'
            report['training_ready'] = False
            report['not_training_ready_reason'] = 'Policy evaluation is not an operator-reviewed demonstration.'
            _save(self.root / 'continuous/report.json', report)
            if self._closed_receipt is None:
                receipt = self.storage.frame_spool.close()
                receipt.update(schema='fr5-stream-closed-v1', codec_sha256=self.storage.codec_sha256,
                               protocol_sha256=self.storage.protocol_sha256, local_episode=str(self.root))
                self._closed_receipt = receipt
            receipt = self._closed_receipt
            _save(self.root / 'stream-closed.json', receipt)
            self._summary['continuous_capture'] = report
            self._summary['storage_status'] = 'server_stream_closed'
            if not report.get('capture_complete'):
                self._summary['outcome'] = 'interrupted'
                if not self._summary['failed']:
                    self._summary['failed'] = report.get('error') or 'Continuous capture incomplete'
                    self._summary['error'] = self._summary['failed']
            self._summary['summary_stage'] = 'final_control_snapshot'
            _save(self.root / 'summary.json', self._summary)
            self.storage.sync_tree()
            self._mark_finalized()
            self._finalized = True
            self._finalize_error = None
            return deepcopy(self._summary)

    def materialize(self):
        self.wait_finalized()
        return self.storage.materialize()


def record_readonly_diagnostic(robot, name, checkpoint_sha, *, seconds=3.,
                               task='记录器连接诊断（不执行动作）'):
    """Record a short stream with an already-connected read-only observer.

    The checkpoint SHA is an identifier only: this function neither loads a
    model nor asserts that checkpoint uses the streaming policy contract.
    The caller owns and disconnects the observer.  Once it is disconnected,
    call the returned recorder's ``wait_finalized`` and ``materialize``.
    """
    if not robot.is_connected:
        raise RuntimeError('Connect the read-only observer before recording')
    if not _is_number(seconds) or seconds <= 0:
        raise ValueError('Diagnostic recording duration must be positive and finite')
    root = new_stream_recording_path(name)
    root.mkdir(mode=0o700)
    scope = {'robot_ip': '192.168.58.2', 'dataset_role': 'policy_evaluation',
             'policy_contract': contract(), 'task_instruction': task,
             'checkpoint_sha256': checkpoint_sha, 'model_execution': False,
             'metadata': {'model_execution': False, 'checkpoint_loaded_by_recorder': False,
                          'checkpoint_sha_role': 'identifier_only_not_loaded',
                          'recording_purpose': 'read_only_connection_diagnostic'}}
    _save(root / 'scope.json', scope)
    recorder = StreamEvaluationRecording(root, scope)
    try:
        recorder.start(robot)
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            recorder.assert_healthy()
            time.sleep(min(.05, max(0., until - time.monotonic())))
    except BaseException as exc:
        recorder.finish('readonly_diagnostic_error', error=f'{type(exc).__name__}: {exc}')
        raise
    recorder.finish('readonly_diagnostic_complete')
    return recorder
