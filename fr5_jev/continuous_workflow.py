"""One continuous pick/place episode with model phase decisions and visual checkpoints.

Import and the default CLI are inert. --execute records continuously, runs each
selected phase to completion, and only requests the LLM for visual facts or an
exception. A checkpoint is an agent message, not a per-action user approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import select
import signal
import sys
import time

from .decision import validate_answer
from .policy import assess

PHASE_SKILLS = {
    'approach_clear': 'transfer_empty_tool', 'approach_near': 'align_xy',
    'approach_contact': 'descend_pregrasp', 'close': 'close_gripper',
    'lift_probe': 'lift_verify', 'lift_full': 'lift_verify',
    'return': 'return_place', 'release': 'release_gripper',
    'retreat_probe': 'retreat_tool', 'retreat_clear': 'retreat_tool',
    'retreat_high': 'retreat_tool',
}
AUTOMATIC = frozenset(('approach_clear', 'approach_near', 'lift_probe',
                       'retreat_clear', 'retreat_high'))
VISUAL_FACTS = {
    'approach_contact': ('exterior_fingers_aligned', 'cup_outside_fingers_until_entry'),
    'close': ('cup_between_fingers_externally',),
    'lift_full': ('cup_follow_visually_verified',),
    'return': ('cup_follow_visually_verified',),
    'release': ('cup_supported_at_original_pose',),
    'retreat_probe': ('cup_supported_and_released',),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def selected_for_phase(record, phase, scope_sha):
    if record.get('scope_sha256') != scope_sha or record.get('decision_error'):
        return False
    try:
        provider = record.get('provider', 'emrank')
        answer = validate_answer(record['request'], record['response'], provider=provider)
        request_state = (record['request']['input'][0]['content'][0]['data']
                         if provider == 'openai' else record['request'].get('state'))
        policy = record['policies_frozen_before_inference'][answer['choice']]
        gate = assess(answer, policy, hard_checks={k: True for k in (
            'specific_authorization', 'preflight_passed', 'scene_reviewed',
            'bounded_executor_verified', 'onsite_confirmation')})
        return (request_state == record['state']
                and record['state']['facts']['requested_phase'] == phase
                and answer['choice'] == PHASE_SKILLS[phase] and gate['route'] == 'selected')
    except (KeyError, IndexError, ValueError, TypeError):
        return False


def validate_automatic_review(session, phase, decision):
    review = decision.get('workflow_review', {})
    if (phase not in AUTOMATIC or review.get('source') != 'deterministic_continuous_workflow'
            or review.get('phase') != phase or review.get('proceed') is not True
            or not selected_for_phase(decision, phase, session.scope_sha)):
        raise RuntimeError('Automatic continuation lacks a matching selected phase decision')
    if phase.startswith('retreat'):
        name = review.get('visual_evidence_record', '')
        if not re.fullmatch(r'[a-z0-9_-]+\.json', name):
            raise RuntimeError('Missing prior release verification')
        path = session.root / name
        if sha(path) != review.get('visual_evidence_sha256'):
            raise RuntimeError('Prior release verification changed')
        prior = json.loads(path.read_text())
        visual = prior.get('current_chat_llm_review', {})
        if (prior.get('scope_sha256') != session.scope_sha
                or visual.get('phase') != 'retreat_probe' or visual.get('proceed') is not True
                or visual.get('cup_supported_and_released') is not True
                or not session.released or 'retreat_probe' not in session.completed_stages):
            raise RuntimeError('Release must be visually verified in this completed episode phase')


class ContinuousWorkflow:
    def __init__(self, session, *, observe, request_review, emit=print):
        self.session = session
        self.observe = observe
        self.request_review = request_review
        self.emit = emit
        self.visual_release_record = None
        self.metrics = {'schema': 'fr5-continuous-workflow-v1', 'phases': [],
                        'llm_checkpoints': 0, 'user_action_approval_requests': 0,
                        'outcome': 'running', 'independent_astra_api_calls': 0}

    def publish(self, event):
        self.emit(json.dumps(event, ensure_ascii=False), flush=True)

    def observe_checkpoint(self, directory):
        storage = getattr(self.session, 'storage', None)
        check = getattr(storage, 'ensure_local_capacity', None)
        if callable(check):
            # Native RGB-D scene cache and both review images must fit before
            # observing; the stream/outbox budget is accounted separately.
            check(reserve_bytes=24 * 1024**2)
        result = self.observe(directory)
        if callable(check):
            check()
        return result

    def visual_review(self, phase, decision_id, selected):
        s = self.session
        directory = s.root / ('view-' + phase)
        observation = self.observe_checkpoint(directory)
        image = directory / 'color.jpg'
        request = {'status': 'llm_visual_checkpoint', 'phase': phase,
                   'scope_sha256': s.scope_sha, 'decision_id': decision_id,
                   'observation_sha256': sha(directory / 'observation.json'),
                   'image': str(image), 'image_sha256': sha(image),
                   'required_facts': list(VISUAL_FACTS.get(phase,
                       ('empty_approach_alignment_verified',) if phase == 'approach_near' else
                       ('cup_supported_and_released',) if phase.startswith('retreat') else ())),
                   'decision_exception': not selected,
                   'state': observation['robot_after'], 'created_monotonic': time.monotonic(),
                   'requires_user_action_approval': False}
        self.metrics['llm_checkpoints'] += 1
        self.publish(request)
        answer = self.request_review(request, s)
        if (not isinstance(answer, dict) or answer.get('phase') != phase
                or answer.get('scope_sha256') != s.scope_sha
                or answer.get('image_sha256') != request['image_sha256']
                or answer.get('proceed') is not True or not answer.get('reason')
                or not 0 <= time.monotonic() - request['created_monotonic'] <= 300
                or any(answer.get(key) is not True for key in request['required_facts'])
                or (not selected and answer.get('resolve_decision_exception') is not True)):
            raise RuntimeError('Visual checkpoint unresolved; no following action dispatched')
        # A review cannot silently survive an intervening jog or a changed image.
        current = s.state(held=phase in ('lift_probe', 'lift_full', 'return', 'release'))
        if (max(abs(a-b) for a, b in zip(current['tcp'], observation['robot_after']['tcp'])) > .1
                or sha(image) != request['image_sha256']
                or sha(directory / 'observation.json') != request['observation_sha256']):
            raise RuntimeError('Checkpoint evidence or robot pose changed')
        review = {key: answer[key] for key in request['required_facts']}
        review.update(phase=phase, proceed=True, source='current_codex_chat',
                      reason=answer['reason'], image=str(image), image_sha256=sha(image),
                      observation_sha256=request['observation_sha256'],
                      reviewed_at_unix=time.time(), reviewed_at_monotonic=time.monotonic(),
                      decision_exception_resolved=not selected)
        if phase.startswith('retreat'):
            review['cup_supported_and_released'] = True
        return review

    def run(self):
        s = self.session
        (s.root / 'WORKFLOW_STARTED').open('x').close()
        finalization_started = False
        try:
            for phase in PHASE_SKILLS:
                started = time.monotonic()
                if getattr(s, 'scope', {}).get('prealigned_start') is True and phase in tuple(PHASE_SKILLS)[:3]:
                    s.record_prealigned_entry(phase)
                    self.publish({'status':'prealigned_start_no_command','phase':phase})
                    continue
                decision_id = 'workflow_' + phase + '.json'
                decision_summary = s.decide(phase, decision_id)
                path = s.root / decision_id
                record = json.loads(path.read_text())
                selected = selected_for_phase(record, phase, s.scope_sha)
                visual_alignment = (phase == 'approach_near'
                                    and getattr(s, 'scope', {}).get('requires_runtime_visual_alignment') is True)
                if phase in AUTOMATIC and selected and not visual_alignment:
                    review = {'source': 'deterministic_continuous_workflow', 'phase': phase,
                              'proceed': True, 'reviewed_at_unix': time.time(),
                              'reviewed_at_monotonic': time.monotonic(),
                              'basis': 'matching selected phase, original route scope and current executor telemetry'}
                    if phase.startswith('retreat'):
                        review.update(cup_supported_and_released=True,
                                      visual_evidence_record=self.visual_release_record,
                                      visual_evidence_sha256=sha(s.root / self.visual_release_record))
                    record['workflow_review'] = review
                    validate_automatic_review(s, phase, record)
                else:
                    record['current_chat_llm_review'] = self.visual_review(phase, decision_id, selected)
                save(path, record)
                (s.root / ('PHASE_' + phase)).open('x').close()
                s.review_phase(phase, decision_id)
                s.run_phase()  # All actions within the phase, with continuous telemetry checks.
                if phase == 'retreat_probe':
                    self.visual_release_record = decision_id
                row = {'phase': phase, 'elapsed_s': time.monotonic()-started,
                       'provider': record.get('provider', 'emrank'),
                       'decision_latency_ms': decision_summary.get('latency_ms'),
                       'automatic_continuation': 'workflow_review' in record,
                       'commands_recorded': s.command_sequence}
                self.metrics['phases'].append(row)
                save(s.root / 'workflow-metrics.json', self.metrics)
                self.publish({'status': 'phase_completed', **row})
            final = self.observe_checkpoint(s.root / 'view-complete')
            evidence = [str(s.root / 'view-complete' / 'color.jpg'),
                        str(s.root / 'workflow-metrics.json')]
            self.metrics['outcome'] = 'pending_user_whole_episode_review'
            save(s.root / 'workflow-metrics.json', self.metrics)
            finalization_started = True
            summary = s.finish('pending_review', evidence)
            self.publish({'status': 'whole_episode_pending_user_review', 'summary': summary,
                          'final_state': final['robot_after'], 'success_count_increment': 0})
            return summary
        except BaseException as exc:
            self.metrics['outcome'] = 'interrupted'
            self.metrics['error_type'] = type(exc).__name__
            try:
                # Once finalization starts, keep the snapshotted metrics
                # immutable even if a later remote receipt is unavailable.
                if finalization_started:
                    save(s.root / 'workflow-finalization-error.json', self.metrics)
                else:
                    save(s.root / 'workflow-metrics.json', self.metrics)
            except OSError as log_error:
                # Failure to persist a diagnostic must not mask the original
                # exception or prevent the caller's physical stop path.
                self.metrics['diagnostic_persistence_error'] = repr(log_error)
            raise


def stdin_review(request, session):
    """Agent replies over stdin; no action is queued while a review is pending."""
    while time.monotonic() - request['created_monotonic'] <= 300:
        session.state(held=request['phase'] in ('lift_probe', 'lift_full', 'return', 'release'))
        ready, _, _ = select.select([sys.stdin], [], [], .1)
        if ready:
            line = sys.stdin.readline()
            if not line:
                raise EOFError('Agent review channel closed')
            return json.loads(line)
    raise TimeoutError('Visual review expired')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    from .teacher_varied import VariedSession
    from .observe import capture
    from fr5_runtime.robot import use_shared_state_stream
    class VisualWorkflowSession(VariedSession):
        supports_visual_checkpoints = True
    session = VisualWorkflowSession(args.directory)
    if not args.execute:
        print(json.dumps({'status': 'validated_no_execution', 'phases': list(PHASE_SKILLS),
                          'visual_checkpoints': list(VISUAL_FACTS),
                          'planned_commands': len(session.planned_actions)}))
        return
    signal.signal(signal.SIGINT, lambda *_: session.interrupted.set())
    signal.signal(signal.SIGTERM, lambda *_: session.interrupted.set())
    completed = False
    try:
        session.start()
        with use_shared_state_stream(session.robot.stream):
            ContinuousWorkflow(session, observe=capture, request_review=stdin_review).run()
            completed = True
    except BaseException as exc:
        if session.storage is not None or session.active_command:
            session.stop_once(exc, arm_in_flight=session.active_command)
        raise
    finally:
        try:
            if not completed and session.writer and not session.writer._finished:
                session.finish('interrupted')
        finally:
            if session.robot.is_connected:
                session.robot.disconnect()


if __name__ == '__main__':
    main()
