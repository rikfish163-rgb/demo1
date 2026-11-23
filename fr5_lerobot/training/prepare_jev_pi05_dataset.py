"""Prepare reviewed fr5_jev demonstrations for pi0.5.

The fixed reference keeps source 7 as the old holdout. Additional reviewed
recordings retain their requested train or validation role. The training
dataset contains only training sources; the second dataset includes holdouts
for the project's source-aware trainer.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / 'fr5_lerobot/src'
for path in (PROJECT_ROOT, PROJECT_SRC):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fr5_jev.continuous_workflow import PHASE_SKILLS
from fr5_jev.decision import validate_answer
from lerobot_robot_fairino import export_stream_policy as stream
from lerobot_robot_fairino.feedback import gripper_feedback_valid
from lerobot_robot_fairino.whole_task_source import _verify_control
from lerobot_camera_gemini.timing import MONOTONIC, camera_clock, command_clock, finite_time


DATA_ROOT = Path('/home/data4/hts/demo1/fr5-learning')
REFERENCE = DATA_ROOT / 'datasets/fr5-joint-stream-20261102-v1'
OPENAI_PHASE_DECISION = {'provider': 'openai', 'endpoint': 'https://api.openai.com/v1/decisions',
                         'model': 'gpt-6-luna'}


def reference_sources(reference):
    """Resolve the seven training sources and the fixed original holdout."""
    proof = stream.read_json(Path(reference) / 'meta/stream_provenance.json')
    sources = proof.get('sources', [])
    if (proof.get('schema') != stream.SCHEMA or proof.get('training_data_kind') != 'captured'
            or proof.get('source_count') != 8 or len(sources) != 8
            or [s.get('source_index') for s in sources] != list(range(8))
            or [s.get('split') for s in sources] != ['train'] * 7 + ['validation']):
        raise ValueError('Reference must retain the seven original training sources and source 7 holdout')
    episodes = proof.get('episodes', [])
    if (not episodes or [e.get('episode_index') for e in episodes] != list(range(len(episodes)))
            or any(e.get('source_index') not in range(8)
                            or e.get('split') != sources[e['source_index']]['split'] for e in episodes)
            or [e['episode_index'] for e in episodes if e['split'] == 'train']
                != proof.get('train_episode_indices')
            or [e['episode_index'] for e in episodes if e['split'] == 'validation']
                != proof.get('validation_episode_indices')
            or {e['source_index'] for e in episodes} != set(range(8))):
        raise ValueError('Reference episodes do not preserve the physical-source split')
    return sources


def _matches_reference(source, record):
    materialization = source['materialization']
    return (str(source['root']) == str(Path(record['trial']).resolve())
            and source['source_scope_sha256'] == record['source_scope_sha256']
            and source['source_summary_sha256'] == record['source_summary_sha256']
            and source['review_sha256'] == record['user_review_sha256']
            and materialization['session'] == record['session']
            and materialization['source_stream_sha256'] == record['source_stream_sha256'])


def _synthetic_marker(value):
    if not isinstance(value, dict):
        return False
    for key in ('synthetic', 'is_synthetic', 'simulated', 'is_simulated', 'mock', 'fake'):
        flag = value.get(key)
        if flag is True or type(flag) is int and flag == 1 or (
                isinstance(flag, str) and flag.strip().lower() in ('true', 'yes', '1')):
            return True
    return 'synthetic_id' in value


def _validate_initial_holds(captures, clips):
    """An unissued hold is usable only where real feedback confirms it."""
    checked = 0
    for clip in clips:
        for row in clip:
            mapping = row['mapping']
            target = mapping['gripper_target']
            if target['executed'] is not False:
                continue
            if target['source'] != 'initial_hold_not_rpc':
                raise ValueError('Unissued gripper target is not a recorded initial hold')
            percent = target['target_degrees_percent'][6]
            indexes = (mapping['observation']['frame_index'],
                       mapping['joint_retarget']['lower']['frame_index'])
            for index in indexes:
                robot = captures[index]['robot']
                if (not gripper_feedback_valid(robot)
                        or abs(float(robot['gripper_position']) - float(percent)) > 1e-6):
                    raise ValueError('Initial gripper hold is not supported by valid recorded feedback')
            checked += 1
    return checked


def _validate_openai_record(teacher, decision, *, image_bytes=None):
    phase, cycle = teacher.get('phase'), teacher.get('cycle')
    if (decision.get('provider') != OPENAI_PHASE_DECISION['provider']
            or decision.get('endpoint') != OPENAI_PHASE_DECISION['endpoint']
            or decision.get('requested_model') != OPENAI_PHASE_DECISION['model']
            or decision.get('selected_by') != 'openai_decisions' or decision.get('decision_error')):
        raise ValueError('Recorded OpenAI phase decision does not match the source contract')
    state = decision.get('state')
    if (_synthetic_marker(decision) or _synthetic_marker(state)
            or not isinstance(state, dict) or state.get('source') != 'live_readonly'
            or _synthetic_marker(state.get('robot')) or _synthetic_marker(state.get('camera'))):
        raise ValueError('Recorded OpenAI phase decision lacks captured state evidence')
    facts = state.get('facts')
    if (not isinstance(facts, dict) or facts.get('requested_phase') != phase
            or type(facts.get('cycle')) is not int or facts['cycle'] != cycle):
        raise ValueError('Recorded OpenAI state differs from the teacher phase and cycle')
    request, response = decision.get('request'), decision.get('response')
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ValueError('Recorded OpenAI phase decision lacks its request and response')
    content = [{'type': 'input_json', 'data': state}]
    if image_bytes is not None:
        inputs = request.get('input')
        if (not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict)
                or not isinstance(inputs[0].get('content'), list) or len(inputs[0]['content']) != 2):
            raise ValueError('Recorded OpenAI observation request lacks its JSON and JPEG input')
        image_input = inputs[0]['content'][1]
        if (not isinstance(image_input, dict) or set(image_input) != {'type', 'image_url'}
                or image_input.get('type') != 'input_image'
                or not isinstance(image_input.get('image_url'), str)
                or not image_input['image_url'].startswith('data:image/jpeg;base64,')):
            raise ValueError('Recorded OpenAI observation input is not an inline JPEG')
        try:
            supplied = base64.b64decode(image_input['image_url'].split(',', 1)[1], validate=True)
        except (binascii.Error, ValueError):
            raise ValueError('Recorded OpenAI observation input has invalid JPEG base64') from None
        if supplied != image_bytes:
            raise ValueError('Recorded OpenAI input image differs from the persisted observation JPEG')
        content.append(image_input)
    if request.get('input') != [{'role': 'user', 'content': content}]:
        raise ValueError('Recorded OpenAI request input differs from the recorded state')
    answer = validate_answer(request, response, provider='openai')
    instructions = request['questions'][0].get('instructions')
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError('Recorded OpenAI next_skill question lacks instructions')
    if answer != decision.get('answer'):
        raise ValueError('Recorded OpenAI internal answer differs from its response')
    return answer


def _validate_scene_observation(decision, captures, previous_completion):
    state = decision['state']
    observation = state['facts'].get('scene_observation')
    if not isinstance(observation, dict) or _synthetic_marker(observation):
        raise ValueError('Recorded OpenAI decision lacks a captured scene observation')
    index = observation.get('frame_index')
    if type(index) is not int or not 0 <= index < len(captures):
        raise ValueError('Recorded scene observation has an invalid capture frame index')
    frame = captures[index]
    if (frame['frame_index'] != index or observation.get('camera') != frame['camera']
            or observation.get('robot') != frame['robot']):
        raise ValueError('Recorded scene observation differs from its paired capture frame')
    captured, basis = camera_clock(frame['camera'])
    robot_time = finite_time(frame['robot'].get('received_monotonic'))
    if (basis != MONOTONIC or observation.get('clock') != MONOTONIC
            or observation.get('camera_time') != captured
            or observation.get('robot_time_monotonic') != robot_time
            or observation.get('previous_completion_monotonic') != previous_completion
            or (previous_completion is not None and min(captured, robot_time) <= previous_completion)):
        raise ValueError('Recorded scene observation clocks do not follow the previous completed command')
    camera, robot = frame['camera'], frame['robot']
    snapshot_camera, snapshot_robot = state.get('camera', {}), state['robot']
    expected_camera = {'serial': camera['source_camera_serial'],
                       'capture_time_unix': camera['source_timestamp_unix'],
                       'capture_time_monotonic': captured}
    expected_robot = {key: robot[key] for key in
                      ('gripper_position', 'gripper_motiondone', 'gripper_active')}
    expected_robot.update(received_monotonic=robot_time,
                          gripper_feedback_valid=gripper_feedback_valid(robot))
    if (not isinstance(snapshot_camera, dict) or not isinstance(snapshot_robot, dict)
            or state.get('captured_at_unix') != camera['source_timestamp_unix']
            or any(snapshot_camera.get(key) != value for key, value in expected_camera.items())
            or any(snapshot_robot.get(key) != value for key, value in expected_robot.items())
            or 'tcp' not in robot or state['facts'].get('current_tcp_mm_deg') != robot['tcp']):
        raise ValueError('Recorded OpenAI state differs from its captured scene observation')
    return observation


def _validate_openai_phase_decision(teacher, name, decision, source, manifest, captures, journal):
    phase, cycle = teacher.get('phase'), teacher.get('cycle')
    if not isinstance(phase, str) or phase not in PHASE_SKILLS or type(cycle) is not int or cycle < 1:
        raise ValueError('Recorded OpenAI decision filename differs from the teacher phase and cycle')
    base_name = f'jev-c{cycle:02d}-{phase}.json'
    followup_name = f'jev-c{cycle:02d}-{phase}-after-observe.json'
    if name not in (base_name, followup_name):
        raise ValueError('Recorded OpenAI decision filename differs from the teacher phase and cycle')
    first_issue = min(command_clock(row, MONOTONIC) for row in journal
                      if row.get('executed') is True and row.get('teacher', {}).get('decision_id') == name)
    prior = [row for segment in source['segments'] for row in segment
             if row['command_issued_monotonic'] < first_issue]
    previous_completion = (max(prior, key=lambda row: row['command_issued_monotonic'])[
                           'command_completed_monotonic'] if prior else None)
    if name == base_name:
        answer = _validate_openai_record(teacher, decision)
        if 'scene_observation' in decision['state']['facts']:
            observation = _validate_scene_observation(decision, captures, previous_completion)
            started = finite_time(decision.get('request_started_monotonic'))
            received = finite_time(decision.get('response_received_monotonic'))
            if not max(observation['camera_time'], observation['robot_time_monotonic']) \
                    <= started <= received <= first_issue:
                raise ValueError('Recorded OpenAI initial observation and decision clocks are not causally ordered')
    else:
        root = source['root']
        base = stream.read_json(_verify_control(root, manifest, base_name))
        if base.get('scope_sha256') != source['source_scope_sha256']:
            raise ValueError('Recorded OpenAI base phase decision does not match the source')
        base_answer = _validate_openai_record(teacher, base)
        if base_answer['choice'] != 'observe_scene':
            raise ValueError('Recorded OpenAI base response did not request observe_scene')
        observation_name = f'observe-c{cycle:02d}-{phase}.json'
        image_name = f'observe-c{cycle:02d}-{phase}.jpg'
        if (base.get('route') != 'observe_scene' or base.get('automatic_continuation') is not False
                or base.get('observation_error') or base.get('observation_record') != observation_name
                or base.get('followup_decision_id') != name
                or decision.get('automatic_continuation') is not True):
            raise ValueError('Recorded OpenAI observe_scene continuation chain is inconsistent')
        observation = stream.read_json(_verify_control(root, manifest, observation_name))
        state = decision.get('state')
        if not isinstance(state, dict) or not isinstance(state.get('facts'), dict):
            raise ValueError('Recorded OpenAI phase decision lacks captured state evidence')
        if (observation.get('record') != observation_name or observation.get('image_file') != image_name
                or observation.get('requested_by_decision') != base_name
                or state['facts'].get('scene_observation') != observation):
            raise ValueError('Recorded OpenAI observation file differs from the decision state or filename')
        image_path = _verify_control(root, manifest, image_name)
        image_bytes = image_path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.format != 'JPEG':
                    raise ValueError('Recorded observation image is not a JPEG')
                image.load()
        except OSError:
            raise ValueError('Recorded observation image is not a readable JPEG') from None
        answer = _validate_openai_record(teacher, decision, image_bytes=image_bytes)
        base_observation = _validate_scene_observation(base, captures, previous_completion)
        _validate_scene_observation(decision, captures, previous_completion)
        started = finite_time(base.get('request_started_monotonic'))
        received = finite_time(base.get('response_received_monotonic'))
        requested = finite_time(observation.get('requested_monotonic'))
        followup_started = finite_time(decision.get('request_started_monotonic'))
        followup_received = finite_time(decision.get('response_received_monotonic'))
        if (not max(base_observation['camera_time'], base_observation['robot_time_monotonic'])
                    <= started <= received <= requested
                or observation['frame_index'] <= base_observation['frame_index']
                or min(observation['camera_time'], observation['robot_time_monotonic']) <= requested
                or not max(observation['camera_time'], observation['robot_time_monotonic'])
                    <= followup_started <= followup_received <= first_issue):
            raise ValueError('Recorded OpenAI observation and decision clocks are not causally ordered')
        frame = captures[observation['frame_index']]
        recorded_image = root / 'continuous' / frame['rgb_file']
        if image_bytes != recorded_image.read_bytes():
            raise ValueError('Recorded observation JPEG differs from its captured frame image')
    if answer['choice'] != PHASE_SKILLS[phase]:
        raise ValueError('Recorded OpenAI choice differs from the executed teacher phase')
    if (decision.get('route') in ('observe_scene', 'needs_llm') or decision.get('observation_error')
            or decision.get('automatic_continuation', True) is not True):
        raise ValueError('Recorded OpenAI phase decision did not permit the executed teacher phase')


def inspect_source(source):
    """Enforce captured fr5_jev evidence in addition to the exporter checks."""
    provenance = source['data_provenance']
    if (provenance.get('training_data_kind') != 'captured'
            or provenance.get('scope_contract') != 'fr5-whole-task-collection-v1'
            or provenance.get('scope_markers') or provenance.get('camera_markers')
            or not provenance.get('camera_serials')):
        raise ValueError('Source is not a captured fr5_jev whole-task recording')
    captures, journal, _ = stream._source_data(source)
    if (_synthetic_marker(source['scope'])
            or any(_synthetic_marker(frame) or _synthetic_marker(frame.get('robot'))
                   or _synthetic_marker(frame.get('camera')) or _synthetic_marker(frame.get('media_codec'))
                   for frame in captures)
            or any(_synthetic_marker(command) for command in journal)):
        raise ValueError('Synthetic or simulated source evidence is not training data')
    manifest = stream.read_json(source['root'] / 'control-manifest.json')
    openai_source = 'phase_decision' in source['scope']
    if openai_source and source['scope']['phase_decision'] != OPENAI_PHASE_DECISION:
        raise ValueError('Source scope phase decision differs from the OpenAI contract')
    policy = 'current_codex_chat+gpt-6-luna' if openai_source else 'current_codex_chat+jev'
    decisions = set()
    for command in journal:
        teacher = command.get('teacher', {})
        if command.get('source') != 'agent_teacher' or teacher.get('policy') != policy:
            raise ValueError('Command journal is not from the fr5_jev teacher')
        if command.get('executed') is not True:
            continue
        name = teacher.get('decision_id')
        if name == 'scope.json' and teacher.get('phase') == 'initial_open':
            continue
        if (not isinstance(name, str) or Path(name).name != name
                or not name.startswith(('jev-c', 'llm-c')) or name not in manifest.get('files', {})):
            raise ValueError('Executed command lacks its recorded fr5_jev phase decision')
        decision = stream.read_json(_verify_control(source['root'], manifest, name))
        if (decision.get('scope_sha256') != source['source_scope_sha256']
                or not decision.get('selected_by')):
            raise ValueError('Recorded phase decision does not match the source')
        if openai_source:
            _validate_openai_phase_decision(teacher, name, decision, source, manifest, captures, journal)
        if name.startswith('jev-c'):
            decisions.add(name)
    if not decisions:
        raise ValueError('Source has no recorded Jev phase decision')
    clips, coverage = stream._build_clips(source, allow_initial_hold=True)
    hold_count = _validate_initial_holds(captures, clips)
    if hold_count != coverage['initial_hold_action_frames']:
        raise ValueError('Initial hold accounting differs from exported rows')
    return {'trial': str(source['root']), 'session': source['materialization']['session'],
            'camera_serials': provenance['camera_serials'], 'user_verdict': source['review']['verdict'],
            'task_instruction': source['task_instruction'], 'coverage': coverage,
            'initial_hold_feedback_verified_frames': hold_count}


def inspect(reference=REFERENCE, *, train_sources=(), validation_sources=()):
    records = reference_sources(reference)
    reference_paths = [Path(r['trial']).resolve() for r in records]
    train_paths = [Path(path).resolve() for path in train_sources]
    validation_paths = [Path(path).resolve() for path in validation_sources]
    paths = reference_paths + train_paths + validation_paths
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate source path')
    sources = [stream.load_source(path) for path in paths]
    stream._check_sources(sources)
    if any(not _matches_reference(source, record) for source, record in zip(sources[:8], records)):
        raise ValueError('A reference source or user review no longer matches the original dataset')
    additional_roles = (['training_complete_trial'] * len(train_paths)
                        + ['held_out_complete_trial'] * len(validation_paths))
    for source, expected in zip(sources[8:], additional_roles):
        if source['scope'].get('split_role') != expected:
            raise ValueError(f"Source {source['root']} must have scope split_role {expected}")
    details = [inspect_source(source) for source in sources]
    train = list(range(7)) + list(range(8, 8 + len(train_paths)))
    heldout = [7] + list(range(8 + len(train_paths), len(paths)))
    for index, (source, detail) in enumerate(zip(sources, details)):
        detail.update({'source_index': index, 'split': 'validation' if index in heldout else 'train',
                       'scope_split_role': source['scope'].get('split_role'),
                       'object_description': source['task_spec'].get('object_description')})
    return {'reference_dataset': str(Path(reference).resolve()),
            'train_source_indices': train, 'holdout_source_indices': heldout,
            'sources': details,
            'training_frames': sum(details[i]['coverage']['exported_frames'] for i in train),
            'holdout_frames': sum(details[i]['coverage']['exported_frames'] for i in heldout),
            'label_semantics': 'future measured joint feedback; issued gripper target or feedback-verified initial hold',
            'provenance_limit': 'recorded source evidence and camera metadata, not hardware attestation'}


def prepare(output, *, reference=REFERENCE, train_sources=(), validation_sources=()):
    output = Path(output).resolve()
    if not output.is_relative_to(DATA_ROOT.resolve()) or output == DATA_ROOT.resolve():
        raise ValueError('Output must be a new directory inside data4/fr5-learning')
    if output.exists():
        raise FileExistsError('Output already exists')
    reference_path = Path(reference).resolve()
    if output.is_relative_to(reference_path) or reference_path.is_relative_to(output):
        raise ValueError('Output and reference dataset directories must not overlap')
    report = inspect(reference, train_sources=train_sources, validation_sources=validation_sources)
    paths = [Path(item['trial']) for item in report['sources']]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in paths):
        raise ValueError('Output and source directories must not overlap')
    output.mkdir(parents=True, exist_ok=False)
    train_paths = [paths[i] for i in report['train_source_indices']]
    train_result = stream.export(train_paths, output / 'train', validation_indices=[], allow_initial_hold=True)
    combined_result = stream.export(paths, output / 'with-validation',
                                    validation_indices=report['holdout_source_indices'],
                                    allow_initial_hold=True)
    report.update({'train_dataset': str(output / 'train'),
                   'source_aware_dataset': str(output / 'with-validation'),
                   'train_export': train_result, 'source_aware_export': combined_result,
                   'source_aware_loader_note': 'Use provenance splits; a default LeRobot loader includes every episode.'})
    (output / 'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                            encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new output directory inside data4/fr5-learning')
    parser.add_argument('--train-source', type=Path, action='append', default=[],
                        help='additional reviewed materialized training directory; repeatable')
    parser.add_argument('--validation-source', type=Path, action='append', default=[],
                        help='additional reviewed materialized held-out directory; repeatable')
    args = parser.parse_args()
    options = {'train_sources': args.train_source, 'validation_sources': args.validation_source}
    report = prepare(args.output, **options) if args.output else inspect(**options)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
