"""Tests for fr5_jev provenance, feedback-backed labels, and held-out sources."""
import base64
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
prepare = load_module('prepare_jev_pi05_dataset', HERE.parent / 'training/prepare_jev_pi05_dataset.py')
fixtures = load_module('stream_data_fixtures', HERE / 'stream_fixtures.py')
from lerobot_robot_fairino import whole_task_source
from fr5_lerobot.tests.test_whole_task_source import set_fixture_scope
from fr5_jev.decision import CONTRACT_VERSION, make_request


def valid_gripper(position=100):
    return {'gripper_position': position, 'gripper_active': 1, 'gripper_fault': 0,
            'gripper_fault_id': 0, 'gripper_motiondone': 1, 'gripper_speed': 0}


def update_fixture_manifest(root, *paths):
    manifest_path = root / 'control-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for path in paths:
        manifest['files'][str(path.relative_to(root))] = {
            'bytes': path.stat().st_size, 'sha256': prepare.stream.sha(path)}
    fixtures.write_json(manifest_path, manifest)


def write_fixture_journal(root, journal):
    journal_path = root / 'target-command-journal.json'
    fixtures.write_json(journal_path, journal)
    commands = {command['sequence']: command for command in journal}
    command_paths = sorted(root.glob('command-*.json'))
    for path in command_paths:
        wire = json.loads(path.read_text())
        wire['command'] = commands[wire['command']['sequence']]
        wire['phase'] = wire['command']['teacher']['phase']
        fixtures.write_json(path, wire)
    teachers_path = root / 'lerobot/meta/capture.jsonl'
    teachers = prepare.stream.read_jsonl(teachers_path)
    for teacher in teachers:
        teacher['command'] = commands[teacher['command']['sequence']]
    fixtures.write_jsonl(teachers_path, teachers)
    material_path = root / 'stream-materialization.json'
    material = json.loads(material_path.read_text())
    material['teacher_journal_sha256'] = prepare.stream.sha(teachers_path)
    fixtures.write_json(material_path, material)
    update_fixture_manifest(root, journal_path, *command_paths)


def make_jev_file_fixture(root, trial, role, description, *, provider='emrank', first_gripper_command=False):
    """Build a temporary simulated recording with Jev or official decision evidence."""
    instruction = f'抓取桌面上的{description}，在空中维持5秒后放下'
    fixtures.make_source(root, trial=trial, instruction=instruction,
                         first_gripper_command=first_gripper_command)
    (root / 'TEST_FIXTURE_ONLY').write_text('Simulated test fixture; not a physical demonstration.\n')
    scope_path = root / 'scope.json'
    scope = json.loads(scope_path.read_text())
    del scope['synthetic_id']
    scope['contract'] = 'fr5-whole-task-collection-v1'
    scope['initial_tcp'] = [float(trial), 0., 200., 0., 0., 0.]
    if provider == 'openai':
        scope['phase_decision'] = dict(prepare.OPENAI_PHASE_DECISION)
    fixtures.write_json(scope_path, scope)
    set_fixture_scope(root, role, object_description=description)
    review_path = root / 'user-review.json'
    review = json.loads(review_path.read_text())
    review['user_statement'] = 'Temporary test fixture review; no physical demonstration performed.'
    fixtures.write_json(review_path, review)

    journal_path = root / 'target-command-journal.json'
    journal = json.loads(journal_path.read_text())
    decisions = {}
    for command in journal:
        phase = command.get('teacher', {}).get('phase', 'initial_hold')
        decision = (f'jev-c01-{phase}.json' if command['executed'] and phase != 'initial_open'
                    else 'scope.json')
        command['source'] = 'agent_teacher'
        policy = 'current_codex_chat+gpt-6-luna' if provider == 'openai' else 'current_codex_chat+jev'
        command['teacher'] = {'policy': policy, 'phase': phase,
                              'cycle': 1, 'decision_id': decision}
        if command['executed'] and decision != 'scope.json':
            record = {'scope_sha256': prepare.stream.sha(scope_path), 'selected_by': 'jev'}
            if provider == 'openai':
                skill = prepare.PHASE_SKILLS[phase]
                state = {'schema': CONTRACT_VERSION, 'source': 'live_readonly', 'robot_id': 'J1',
                         'robot_ip': '192.168.58.2', 'captured_at_unix': command['issued_unix'],
                         'robot': {'stationary': True, 'fault_free': True},
                         'candidates': {skill: {'validated': True, 'preconditions_met': True,
                                                'evidence': 'Temporary test fixture only'}},
                         'facts': {'requested_phase': phase, 'cycle': 1}}
                request = make_request(state, provider='openai')
                probabilities = {entry['value']: float(entry['value'] == skill)
                                 for entry in request['questions'][0]['choices']}
                answer = {'type': 'choice', 'choice': skill, 'confidence': 1., 'probabilities': probabilities}
                response = {'model': 'gpt-6-luna', 'answers': [
                    {**answer, 'name': 'next_skill',
                     'probabilities': [{'value': value, 'probability': probability}
                                       for value, probability in probabilities.items()]}]}
                record.update(provider='openai', endpoint='https://api.openai.com/v1/decisions',
                              requested_model='gpt-6-luna', selected_by='openai_decisions',
                              state=state, request=request, response=response, answer=answer,
                              test_fixture_only=True)
            decisions[decision] = record
    fixtures.write_json(journal_path, journal)
    for name, decision in decisions.items():
        fixtures.write_json(root / name, decision)

    commands = {command['sequence']: command for command in journal}
    command_paths = sorted(root.glob('command-*.json'))
    for path in command_paths:
        wire = json.loads(path.read_text())
        wire['command'] = commands[wire['command']['sequence']]
        wire['observation_before_command']['source_camera_serial'] = 'fixture-camera'
        fixtures.write_json(path, wire)

    captures_path = root / 'continuous/capture.jsonl'
    teachers_path = root / 'lerobot/meta/capture.jsonl'
    media_path = root / 'stream-frames.jsonl'
    captures = prepare.stream.read_jsonl(captures_path)
    teachers = prepare.stream.read_jsonl(teachers_path)
    media = prepare.stream.read_jsonl(media_path)
    first_gripper = next(command['issued_monotonic'] for command in journal
                         if command['executed'] and command['kind'] == 'gripper')
    for capture, teacher, image in zip(captures, teachers, media):
        camera = capture['camera']
        camera['source_camera_serial'] = 'fixture-camera'
        capture['robot']['gripper_position'] = 100 if camera['source_timestamp_monotonic'] < first_gripper else 82
        teacher['camera'], teacher['robot'] = camera, capture['robot']
        teacher['command'] = commands[teacher['command']['sequence']]
        image['camera'] = camera
    fixtures.write_jsonl(captures_path, captures)
    fixtures.write_jsonl(teachers_path, teachers)
    fixtures.write_jsonl(media_path, media)

    manifest_path = root / 'control-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['session'] = f'fixture-session-{trial}'
    for path in (journal_path, *command_paths, *(root / name for name in decisions)):
        manifest['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': prepare.stream.sha(path)}
    fixtures.write_json(manifest_path, manifest)
    material_path = root / 'stream-materialization.json'
    material = json.loads(material_path.read_text())
    material['session'] = manifest['session']
    material['capture_journal_sha256'] = prepare.stream.sha(captures_path)
    material['teacher_journal_sha256'] = prepare.stream.sha(teachers_path)
    fixtures.write_json(material_path, material)
    return root


def make_fixture_reference(root, sources):
    records, episodes = [], []
    for index, path in enumerate(sources):
        source = prepare.stream.load_source(path)
        split = 'train' if index < 7 else 'validation'
        records.append({'source_index': index, 'split': split, 'trial': str(path),
                        'source_scope_sha256': source['source_scope_sha256'],
                        'source_summary_sha256': source['source_summary_sha256'],
                        'user_review_sha256': source['review_sha256'],
                        'session': source['materialization']['session'],
                        'source_stream_sha256': source['materialization']['source_stream_sha256']})
        episodes.append({'episode_index': index, 'source_index': index, 'split': split})
    (root / 'meta').mkdir(parents=True)
    fixtures.write_json(root / 'meta/stream_provenance.json',
                        {'schema': prepare.stream.SCHEMA, 'training_data_kind': 'captured',
                         'source_count': 8, 'sources': records, 'episodes': episodes,
                         'train_episode_indices': list(range(7)), 'validation_episode_indices': [7]})
    return root


def set_fixture_choice(record, choice):
    wire = record['response']['answers'][0]
    wire['choice'] = choice
    for entry in wire['probabilities']:
        entry['probability'] = float(entry['value'] == choice)
    record['answer'] = {'type': wire['type'], 'choice': choice, 'confidence': wire['confidence'],
                        'probabilities': {entry['value']: entry['probability']
                                          for entry in wire['probabilities']}}


def set_fixture_scene(record, scene):
    state = record['state']
    state['captured_at_unix'] = scene['camera']['source_timestamp_unix']
    state['camera'] = {'serial': scene['camera']['source_camera_serial'], 'available': True,
                       'expected_device_and_fresh': True,
                       'capture_time_unix': scene['camera']['source_timestamp_unix'],
                       'capture_time_monotonic': scene['camera_time']}
    state['robot'].update({key: scene['robot'][key] for key in
                          ('gripper_position', 'gripper_motiondone', 'gripper_active')})
    state['robot'].update(received_monotonic=scene['robot_time_monotonic'],
                          gripper_feedback_valid=prepare.gripper_feedback_valid(scene['robot']))
    state['facts'].update(scene_observation=deepcopy(scene),
                          current_tcp_mm_deg=deepcopy(scene['robot']['tcp']))
    record['request']['input'][0]['content'][0]['data'] = deepcopy(state)
    started = scene['robot_time_monotonic'] + .01
    record.update(request_started_monotonic=started, request_started_unix=started + 1900,
                  response_received_monotonic=started + .01,
                  response_received_unix=started + 1900.01, automatic_continuation=True)


def make_paired_file_fixture(root, trial=0, role='training_complete_trial', description='紫色方块',
                             *, first_gripper_command=False):
    """Build recorded-pair decision files in a temporary simulated source."""
    make_jev_file_fixture(root, trial, role, description, provider='openai',
                          first_gripper_command=first_gripper_command)
    captures_path, teachers_path = root / 'continuous/capture.jsonl', root / 'lerobot/meta/capture.jsonl'
    captures, teachers = prepare.stream.read_jsonl(captures_path), prepare.stream.read_jsonl(teachers_path)
    journal = prepare.stream.read_json(root / 'target-command-journal.json')
    commands = {row['sequence']: prepare.stream.read_json(root / f"command-{row['sequence']:04d}.json")
                for row in journal if row['executed']}
    for frame, teacher in zip(captures, teachers):
        frame['robot']['tcp'] = [-117., -809., 199., float(frame['frame_index']), 0., 0.]
        completed_grippers = [row for row in journal if row['executed'] and row['kind'] == 'gripper'
                              and commands[row['sequence']]['completed_monotonic']
                                  < frame['camera']['source_timestamp_monotonic']]
        closed = completed_grippers and completed_grippers[-1]['teacher']['phase'] == 'close'
        frame['robot'].update(gripper_position=82 if closed else 100,
                               gripper_motiondone=2 if closed else 1)
        teacher['robot'] = frame['robot']
    fixtures.write_jsonl(captures_path, captures)
    fixtures.write_jsonl(teachers_path, teachers)
    material_path = root / 'stream-materialization.json'
    material = prepare.stream.read_json(material_path)
    material['capture_journal_sha256'] = prepare.stream.sha(captures_path)
    material['teacher_journal_sha256'] = prepare.stream.sha(teachers_path)
    fixtures.write_json(material_path, material)

    names = dict.fromkeys(row['teacher']['decision_id'] for row in journal
                          if row['executed'] and row['teacher']['decision_id'] != 'scope.json')
    for name in names:
        issued = min(row['issued_monotonic'] for row in journal if row['teacher']['decision_id'] == name)
        prior = [row for row in journal if row['executed'] and row['issued_monotonic'] < issued]
        completed = commands[prior[-1]['sequence']]['completed_monotonic'] if prior else None
        frame = [frame for frame in captures
                 if (completed is None or min(frame['camera']['source_timestamp_monotonic'],
                                               frame['robot']['received_monotonic']) > completed)
                 and frame['robot']['received_monotonic'] + .03 < issued][-1]
        scene = {'frame_index': frame['frame_index'], 'camera': deepcopy(frame['camera']),
                 'robot': deepcopy(frame['robot']), 'clock': prepare.MONOTONIC,
                 'camera_time': frame['camera']['source_timestamp_monotonic'],
                 'robot_time_monotonic': frame['robot']['received_monotonic'],
                 'previous_completion_monotonic': completed}
        record = prepare.stream.read_json(root / name)
        set_fixture_scene(record, scene)
        fixtures.write_json(root / name, record)
    update_fixture_manifest(root, *(root / name for name in names))
    return root


def make_observe_file_fixture(root, trial=0, role='training_complete_trial', description='紫色方块'):
    """Build an observe/redecision file chain; no physical demonstration is performed."""
    make_paired_file_fixture(root, trial, role, description)
    captures = prepare.stream.read_jsonl(root / 'continuous/capture.jsonl')
    base_name, followup_name = 'jev-c01-release.json', 'jev-c01-release-after-observe.json'
    observation_name, image_name = 'observe-c01-release.json', 'observe-c01-release.jpg'
    journal = prepare.stream.read_json(root / 'target-command-journal.json')
    issued = next(row['issued_monotonic'] for row in journal if row['teacher']['decision_id'] == base_name)
    prior = max((row for row in journal if row['executed'] and row['issued_monotonic'] < issued),
                key=lambda row: row['issued_monotonic'])
    completed = prepare.stream.read_json(root / f"command-{prior['sequence']:04d}.json")['completed_monotonic']
    frames = [frame for frame in captures if frame['camera']['source_timestamp_monotonic'] > completed
              and frame['robot']['received_monotonic'] < issued]
    base_frame, fresh_frame = frames[:2]
    requested = base_frame['robot']['received_monotonic'] + .03
    observation = {'frame_index': fresh_frame['frame_index'], 'camera': deepcopy(fresh_frame['camera']),
                   'robot': deepcopy(fresh_frame['robot']), 'clock': 'host_monotonic',
                   'camera_time': fresh_frame['camera']['source_timestamp_monotonic'],
                   'robot_time_monotonic': fresh_frame['robot']['received_monotonic'],
                   'previous_completion_monotonic': completed, 'requested_by_decision': base_name,
                   'requested_monotonic': requested, 'requested_unix': requested + 1900,
                   'record': observation_name, 'image_file': image_name}
    base_observation = {'frame_index': base_frame['frame_index'], 'camera': deepcopy(base_frame['camera']),
                        'robot': deepcopy(base_frame['robot']), 'clock': 'host_monotonic',
                        'camera_time': base_frame['camera']['source_timestamp_monotonic'],
                        'robot_time_monotonic': base_frame['robot']['received_monotonic'],
                        'previous_completion_monotonic': completed}
    original = prepare.stream.read_json(root / base_name)
    base, followup = deepcopy(original), deepcopy(original)
    for record, scene in ((base, base_observation), (followup, observation)):
        set_fixture_scene(record, scene)
    set_fixture_choice(base, 'observe_scene')
    base.update(route='observe_scene', automatic_continuation=False,
                observation_record=observation_name, followup_decision_id=followup_name)
    followup['automatic_continuation'] = True
    image_path = root / 'continuous' / fresh_frame['rgb_file']
    Image.new('RGB', (640, 360), (120 + trial, 20, 180)).save(image_path, format='JPEG')
    media_path = root / 'stream-frames.jsonl'
    media = prepare.stream.read_jsonl(media_path)
    media[fresh_frame['frame_index']]['rgb_jpeg_sha256'] = prepare.stream.sha(image_path)
    fixtures.write_jsonl(media_path, media)
    image_bytes = image_path.read_bytes()
    (root / image_name).write_bytes(image_bytes)
    followup['request']['input'][0]['content'].append(
        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode()})
    for name, record in ((base_name, base), (followup_name, followup), (observation_name, observation)):
        fixtures.write_json(root / name, record)
    for row in journal:
        if row['teacher']['decision_id'] == base_name:
            row['teacher']['decision_id'] = followup_name
    write_fixture_journal(root, journal)
    update_fixture_manifest(root, *(root / name for name in
                            (base_name, followup_name, observation_name, image_name)))
    return root


class InitialHoldTests(unittest.TestCase):
    def test_unissued_hold_requires_valid_matching_observation_and_pre_target_feedback(self):
        captures = [{'robot': valid_gripper()}, {'robot': valid_gripper()}]
        row = {'mapping': {'observation': {'frame_index': 0},
                           'joint_retarget': {'lower': {'frame_index': 1}},
                           'gripper_target': {'executed': False, 'source': 'initial_hold_not_rpc',
                                              'target_degrees_percent': [0] * 6 + [100]}}}
        self.assertEqual(prepare._validate_initial_holds(captures, [[row]]), 1)
        captures[1]['robot']['gripper_motiondone'] = 0
        with self.assertRaisesRegex(ValueError, 'valid recorded feedback'):
            prepare._validate_initial_holds(captures, [[row]])
        captures[1]['robot'] = valid_gripper(76)
        with self.assertRaisesRegex(ValueError, 'valid recorded feedback'):
            prepare._validate_initial_holds(captures, [[row]])
        row['mapping']['gripper_target']['executed'] = True
        self.assertEqual(prepare._validate_initial_holds(captures, [[row]]), 0)


class SourceTests(unittest.TestCase):
    def test_synthetic_recording_and_failed_or_unreviewed_trials_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='jev-source-test-') as directory:
            source = fixtures.make_source(Path(directory) / 'source')
            loaded = prepare.stream.load_source(source)
            with self.assertRaisesRegex(ValueError, 'captured fr5_jev'):
                prepare.inspect_source(loaded)
            review_path = source / 'user-review.json'
            review = json.loads(review_path.read_text())
            review['verdict'] = 'failure'
            review_path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, 'User success review'):
                prepare.stream.load_source(source)
            review_path.unlink()
            with self.assertRaises(FileNotFoundError):
                prepare.stream.load_source(source)

    def test_captured_metadata_does_not_override_simulated_robot_evidence(self):
        source = {'root': Path('/not-opened'), 'scope': {},
                  'data_provenance': {'training_data_kind': 'captured',
                                      'scope_contract': 'fr5-whole-task-collection-v1',
                                      'scope_markers': {}, 'camera_markers': [], 'camera_serials': ['real-device']}}
        frame = {'robot': {'simulated': True}, 'camera': {}, 'media_codec': {}}
        with patch.object(prepare.stream, '_source_data', return_value=([frame], [], [])):
            with self.assertRaisesRegex(ValueError, 'Synthetic or simulated'):
                prepare.inspect_source(source)

    def test_reference_requires_fixed_whole_source_holdout(self):
        with tempfile.TemporaryDirectory(prefix='jev-split-test-') as directory:
            meta = Path(directory) / 'meta'
            meta.mkdir()
            sources = [{'source_index': i, 'split': 'train' if i < 7 else 'validation'} for i in range(8)]
            proof = {'schema': prepare.stream.SCHEMA, 'training_data_kind': 'captured',
                     'source_count': 8, 'sources': sources,
                     'episodes': [{'episode_index': i, 'source_index': i, 'split': s['split']}
                                  for i, s in enumerate(sources)],
                     'train_episode_indices': list(range(7)), 'validation_episode_indices': [7]}
            path = meta / 'stream_provenance.json'
            path.write_text(json.dumps(proof))
            self.assertEqual(prepare.reference_sources(directory), sources)
            proof['episodes'][7]['source_index'] = 6
            path.write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError, 'physical-source split'):
                prepare.reference_sources(directory)

    def test_reference_cannot_duplicate_a_physical_recording_across_splits(self):
        records = [{'trial': f'/recording/{i}'} for i in range(8)]
        records[7] = {'trial': '/recording/6'}
        sources = []
        for i in range(8):
            sources.append({'root': Path(f'/recording/{i}'), 'source_scope_sha256': f'scope-{i}',
                            'source_summary_sha256': f'summary-{i}', 'review_sha256': f'review-{i}',
                            'materialization': {'session': f'session-{i}', 'source_stream_sha256': f'stream-{i}'}})
        by_path = {str(s['root']): s for s in sources}
        with patch.object(prepare, 'reference_sources', return_value=records), \
                patch.object(prepare.stream, 'load_source', side_effect=lambda p: by_path[str(p)]):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                prepare.inspect('/reference')


class OpenAIPhaseSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='official-decision-file-fixture-')
        self.addCleanup(self.temp.cleanup)
        api = patch('fr5_jev.decision.call_jev', side_effect=AssertionError('File fixtures cannot call an API'))
        api.start()
        self.addCleanup(api.stop)
        self.source = make_jev_file_fixture(Path(self.temp.name) / 'source', 0,
                                           'training_complete_trial', '透明杯子', provider='openai')
        self.name = 'jev-c01-approach_clear.json'
        self.record = json.loads((self.source / self.name).read_text())
        self.scope = json.loads((self.source / 'scope.json').read_text())
        self.journal = prepare.stream.read_json(self.source / 'target-command-journal.json')

    def inspect_fixture(self):
        return prepare.inspect_source(prepare.stream.load_source(self.source))

    def write_decision(self, record, name=None):
        path = self.source / (name or self.name)
        fixtures.write_json(path, record)
        update_fixture_manifest(self.source, path)

    def write_scope(self, scope):
        fixtures.write_json(self.source / 'scope.json', scope)
        set_fixture_scope(self.source, scope['split_role'])
        paths = list(self.source.glob('jev-c*.json'))
        for path in paths:
            record = json.loads(path.read_text())
            record['scope_sha256'] = prepare.stream.sha(self.source / 'scope.json')
            fixtures.write_json(path, record)
        update_fixture_manifest(self.source, *paths)

    def test_official_recorded_source_is_accepted(self):
        result = self.inspect_fixture()
        self.assertGreater(result['initial_hold_feedback_verified_frames'], 0)
        self.assertEqual(self.scope['phase_decision'],
                         {'provider': 'openai', 'endpoint': 'https://api.openai.com/v1/decisions',
                          'model': 'gpt-6-luna'})
        self.assertEqual({command['teacher']['policy'] for command in self.journal},
                         {'current_codex_chat+gpt-6-luna'})
        self.assertEqual(self.record['request']['input'][0]['content'][0]['data'], self.record['state'])
        self.assertIsInstance(self.record['response']['answers'], list)
        self.assertIsInstance(self.record['response']['answers'][0]['probabilities'], list)
        self.assertIsInstance(self.record['answer']['probabilities'], dict)
        self.assertNotIn('scene_observation', self.record['state']['facts'])
        self.assertIn('not a physical demonstration', (self.source / 'TEST_FIXTURE_ONLY').read_text())

    def test_scope_requires_the_exact_official_contract(self):
        contract = self.scope['phase_decision']
        invalid = [None, {}, 'openai', {**contract, 'provider': 'emrank'},
                   {**contract, 'endpoint': 'https://emrank.tech/v1/decisions'},
                   {**contract, 'model': 'gpt-6-luna-other'}, {**contract, 'extra': 'fixture'}]
        for value in invalid:
            with self.subTest(contract=value):
                self.write_scope({**self.scope, 'phase_decision': value})
                with self.assertRaisesRegex(ValueError, 'scope phase decision'):
                    self.inspect_fixture()

    def test_scope_selects_one_policy_for_all_teacher_rows(self):
        for index in range(len(self.journal)):
            with self.subTest(row=index):
                journal = deepcopy(self.journal)
                journal[index]['teacher']['policy'] = 'current_codex_chat+jev'
                write_fixture_journal(self.source, journal)
                with self.assertRaisesRegex(ValueError, 'fr5_jev teacher'):
                    self.inspect_fixture()
        write_fixture_journal(self.source, self.journal)
        self.write_scope({key: value for key, value in self.scope.items() if key != 'phase_decision'})
        with self.assertRaisesRegex(ValueError, 'fr5_jev teacher'):
            self.inspect_fixture()

    def test_record_provider_endpoint_model_selector_and_scope_match(self):
        invalid = [('provider', 'emrank'), ('endpoint', 'https://emrank.tech/v1/decisions'),
                   ('requested_model', 'jev-1.13.0'), ('selected_by', 'jev'),
                   ('selected_by', 'current_codex_chat'), ('selected_by', 'policy_evaluation'),
                   ('scope_sha256', 'different-fixture-scope'), ('decision_error', 'Fixture inference failure')]
        invalid += [(field, None) for field in ('provider', 'endpoint', 'requested_model', 'selected_by')]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                record = deepcopy(self.record)
                if value is None:
                    record.pop(field)
                else:
                    record[field] = value
                self.write_decision(record)
                with self.assertRaisesRegex(ValueError, 'phase decision does not match'):
                    self.inspect_fixture()

    def test_request_and_response_models_are_pinned(self):
        for field in ('request', 'response'):
            for model in (None, 'jev-1.13.0', 'gpt-6-luna-other'):
                with self.subTest(field=field, model=model):
                    record = deepcopy(self.record)
                    if model is None:
                        record[field].pop('model')
                    else:
                        record[field]['model'] = model
                    self.write_decision(record)
                    with self.assertRaisesRegex(ValueError, 'configured model'):
                        self.inspect_fixture()

    def test_recorded_phase_and_cycle_match_the_teacher(self):
        for field, value in (('requested_phase', 'close'), ('requested_phase', None),
                             ('cycle', 2), ('cycle', True), ('cycle', None)):
            with self.subTest(field=field, value=value):
                record = deepcopy(self.record)
                record['state']['facts'][field] = value
                record['request']['input'][0]['content'][0]['data'] = deepcopy(record['state'])
                self.write_decision(record)
                with self.assertRaisesRegex(ValueError, 'state differs from the teacher phase and cycle'):
                    self.inspect_fixture()

    def test_teacher_phase_and_cycle_match_the_filename(self):
        for field, value in (('phase', 'approach_near'), ('cycle', 2), ('cycle', 0), ('cycle', True)):
            with self.subTest(field=field, value=value):
                journal = deepcopy(self.journal)
                command = next(row for row in journal if row['teacher']['decision_id'] == self.name)
                command['teacher'][field] = value
                write_fixture_journal(self.source, journal)
                with self.assertRaisesRegex(ValueError, 'filename differs from the teacher phase and cycle'):
                    self.inspect_fixture()

    def test_official_decisions_require_the_matching_jev_filename(self):
        for name in ('llm-c01-approach_clear.json', 'jev-c02-approach_clear.json',
                     'jev-c01-approach_near.json'):
            with self.subTest(name=name):
                self.write_decision(self.record, name)
                journal = deepcopy(self.journal)
                for command in journal:
                    if command['teacher']['decision_id'] == self.name:
                        command['teacher']['decision_id'] = name
                write_fixture_journal(self.source, journal)
                with self.assertRaisesRegex(ValueError, 'filename differs from the teacher phase and cycle'):
                    self.inspect_fixture()

    def test_choice_matches_the_executed_phase(self):
        record = deepcopy(self.record)
        wire = record['response']['answers'][0]
        wire['choice'] = 'observe_scene'
        for entry in wire['probabilities']:
            entry['probability'] = float(entry['value'] == wire['choice'])
        record['answer'] = {'type': 'choice', 'choice': wire['choice'], 'confidence': wire['confidence'],
                            'probabilities': {entry['value']: entry['probability']
                                              for entry in wire['probabilities']}}
        self.write_decision(record)
        with self.assertRaisesRegex(ValueError, 'choice differs from the executed teacher phase'):
            self.inspect_fixture()

    def test_request_input_matches_the_recorded_state(self):
        different = deepcopy(self.record['state'])
        different['robot']['fault_free'] = False
        for value in (None, [], [{'role': 'user', 'content': [{'type': 'input_json', 'data': different}]}]):
            with self.subTest(input=value):
                record = deepcopy(self.record)
                record['request']['input'] = value
                self.write_decision(record)
                with self.assertRaisesRegex(ValueError, 'request input differs from the recorded state'):
                    self.inspect_fixture()

    def test_real_request_and_response_are_required(self):
        for field in ('request', 'response'):
            for value in (None, {}):
                with self.subTest(field=field, value=value):
                    record = deepcopy(self.record)
                    if value is None:
                        record.pop(field)
                    else:
                        record[field] = value
                    self.write_decision(record)
                    with self.assertRaises(ValueError):
                        self.inspect_fixture()
        self.write_decision({key: self.record[key] for key in
                             ('scope_sha256', 'provider', 'endpoint', 'requested_model', 'selected_by')})
        with self.assertRaisesRegex(ValueError, 'captured state evidence'):
            self.inspect_fixture()

    def test_internal_answer_matches_the_validated_response(self):
        invalid = [('choice', 'hold'), ('confidence', .5), ('type', 'refusal'),
                   ('probabilities', {**self.record['answer']['probabilities'], 'transfer_empty_tool': .5})]
        for field, value in invalid:
            with self.subTest(field=field):
                record = deepcopy(self.record)
                record['answer'][field] = value
                self.write_decision(record)
                with self.assertRaisesRegex(ValueError, 'internal answer differs from its response'):
                    self.inspect_fixture()
        record = deepcopy(self.record)
        record['response']['answers'][0]['confidence'] = .5
        self.write_decision(record)
        with self.assertRaisesRegex(ValueError, 'internal answer differs from its response'):
            self.inspect_fixture()

    def test_official_wire_shape_probabilities_and_refusals_are_validated(self):
        for case in ('answer_map', 'question_map', 'probability_map', 'probability_sum', 'refusal', 'instructions'):
            with self.subTest(case=case):
                record = deepcopy(self.record)
                if case == 'answer_map':
                    record['response']['answers'] = {'next_skill': record['answer']}
                elif case == 'question_map':
                    record['request']['questions'] = {'next_skill': record['request']['questions'][0]}
                elif case == 'probability_map':
                    record['response']['answers'][0]['probabilities'] = record['answer']['probabilities']
                elif case == 'probability_sum':
                    for entry in record['response']['answers'][0]['probabilities']:
                        entry['probability'] *= .5
                elif case == 'refusal':
                    record['response']['refusal'] = 'Test fixture refusal'
                else:
                    record['request']['questions'][0].pop('instructions')
                self.write_decision(record)
                with self.assertRaises(ValueError):
                    self.inspect_fixture()

    def test_synthetic_decision_evidence_is_rejected(self):
        for case in ('record', 'state', 'state_source', 'robot', 'camera'):
            with self.subTest(case=case):
                record = deepcopy(self.record)
                if case == 'record':
                    record['synthetic'] = True
                elif case == 'state':
                    record['state']['synthetic'] = True
                elif case == 'state_source':
                    record['state']['source'] = 'synthetic_test'
                else:
                    record['state'].setdefault(case, {})['simulated'] = True
                record['request']['input'][0]['content'][0]['data'] = deepcopy(record['state'])
                self.write_decision(record)
                with self.assertRaisesRegex(ValueError, 'captured state evidence'):
                    self.inspect_fixture()

    def test_synthetic_pending_failed_and_policy_evaluation_sources_are_rejected(self):
        self.write_scope({**self.scope, 'synthetic': True})
        with self.assertRaisesRegex(ValueError, 'captured fr5_jev|Synthetic or simulated'):
            self.inspect_fixture()
        self.write_scope(self.scope)
        review_path = self.source / 'user-review.json'
        review = json.loads(review_path.read_text())
        for verdict in ('pending', 'failure'):
            with self.subTest(verdict=verdict):
                fixtures.write_json(review_path, {**review, 'verdict': verdict})
                with self.assertRaisesRegex(ValueError, 'User success review'):
                    self.inspect_fixture()
        fixtures.write_json(review_path, review)
        self.write_scope({**self.scope, 'split_role': 'policy_evaluation'})
        with self.assertRaises(ValueError):
            self.inspect_fixture()
        self.write_scope(self.scope)
        for field in ('policy', 'source'):
            with self.subTest(evaluation_field=field):
                journal = deepcopy(self.journal)
                if field == 'policy':
                    journal[0]['teacher']['policy'] = 'policy_evaluation'
                else:
                    journal[0]['source'] = 'policy_evaluation'
                write_fixture_journal(self.source, journal)
                with self.assertRaisesRegex(ValueError, 'fr5_jev teacher'):
                    self.inspect_fixture()
        write_fixture_journal(self.source, self.journal)
        summary_path = self.source / 'summary.json'
        summary = json.loads(summary_path.read_text())
        fixtures.write_json(summary_path, {**summary, 'outcome': 'failed', 'failed': 'Test fixture failure'})
        self.write_scope(self.scope)
        with self.assertRaisesRegex(ValueError, 'Episode is incomplete'):
            self.inspect_fixture()

    def test_initial_open_and_initial_hold_feedback_are_preserved(self):
        source = make_jev_file_fixture(Path(self.temp.name) / 'initial-open', 1,
                                      'training_complete_trial', '透明杯子', provider='openai',
                                      first_gripper_command=True)
        journal = prepare.stream.read_json(source / 'target-command-journal.json')
        opened = next(command for command in journal if command['teacher']['phase'] == 'initial_open')
        self.assertEqual(opened['teacher']['decision_id'], 'scope.json')
        self.assertGreater(prepare.inspect_source(prepare.stream.load_source(source))[
                           'initial_hold_feedback_verified_frames'], 0)
        captures_path, teachers_path = source / 'continuous/capture.jsonl', source / 'lerobot/meta/capture.jsonl'
        captures, teachers = prepare.stream.read_jsonl(captures_path), prepare.stream.read_jsonl(teachers_path)
        captures[0]['robot']['gripper_position'] = 82
        teachers[0]['robot'] = captures[0]['robot']
        fixtures.write_jsonl(captures_path, captures)
        fixtures.write_jsonl(teachers_path, teachers)
        material_path = source / 'stream-materialization.json'
        material = json.loads(material_path.read_text())
        material['capture_journal_sha256'] = prepare.stream.sha(captures_path)
        material['teacher_journal_sha256'] = prepare.stream.sha(teachers_path)
        fixtures.write_json(material_path, material)
        with self.assertRaisesRegex(ValueError, 'valid recorded feedback'):
            prepare.inspect_source(prepare.stream.load_source(source))

    def test_legacy_jev_and_llm_file_compatibility_is_preserved(self):
        source = make_jev_file_fixture(Path(self.temp.name) / 'legacy', 1,
                                      'training_complete_trial', '透明杯子')
        journal = prepare.stream.read_json(source / 'target-command-journal.json')
        old_name, new_name = self.name, self.name.replace('jev-c', 'llm-c', 1)
        decision = prepare.stream.read_json(source / old_name)
        decision['selected_by'] = 'fixture_llm_review'
        fixtures.write_json(source / new_name, decision)
        update_fixture_manifest(source, source / new_name)
        for command in journal:
            if command['teacher']['decision_id'] == old_name:
                command['teacher']['decision_id'] = new_name
        write_fixture_journal(source, journal)
        self.assertGreater(prepare.inspect_source(prepare.stream.load_source(source))[
                           'initial_hold_feedback_verified_frames'], 0)
        for command in journal:
            name = command['teacher']['decision_id']
            if name.startswith('jev-c'):
                replacement = name.replace('jev-c', 'llm-c', 1)
                fixtures.write_json(source / replacement, prepare.stream.read_json(source / name))
                update_fixture_manifest(source, source / replacement)
                command['teacher']['decision_id'] = replacement
        write_fixture_journal(source, journal)
        with self.assertRaisesRegex(ValueError, 'no recorded Jev phase decision'):
            prepare.inspect_source(prepare.stream.load_source(source))


class OpenAIPairedPhaseSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='paired-initial-file-fixture-')
        self.addCleanup(self.temp.cleanup)
        for target in ('fr5_jev.decision.call_jev', 'fr5_jev.decision.api_key'):
            guard = patch(target, side_effect=AssertionError('File fixtures cannot call an API or read credentials'))
            guard.start()
            self.addCleanup(guard.stop)
        self.source = make_paired_file_fixture(Path(self.temp.name) / 'source')
        self.name = 'jev-c01-close.json'
        self.record = prepare.stream.read_json(self.source / self.name)
        self.captures = prepare.stream.read_jsonl(self.source / 'continuous/capture.jsonl')
        self.journal = prepare.stream.read_json(self.source / 'target-command-journal.json')

    def inspect_fixture(self):
        return prepare.inspect_source(prepare.stream.load_source(self.source))

    def write_record(self, record, name=None):
        record['request']['input'][0]['content'][0]['data'] = deepcopy(record['state'])
        path = self.source / (name or self.name)
        fixtures.write_json(path, record)
        update_fixture_manifest(self.source, path)

    def test_paired_initial_decisions_are_accepted_with_recorded_feedback(self):
        result = self.inspect_fixture()
        self.assertGreater(result['initial_hold_feedback_verified_frames'], 0)
        scene = self.record['state']['facts']['scene_observation']
        frame = self.captures[scene['frame_index']]
        self.assertEqual(scene['camera'], frame['camera'])
        self.assertEqual(scene['robot'], frame['robot'])
        self.assertEqual(self.record['state']['captured_at_unix'], frame['camera']['source_timestamp_unix'])
        self.assertEqual(len(self.record['request']['input'][0]['content']), 1)
        self.assertIn('not a physical demonstration', (self.source / 'TEST_FIXTURE_ONLY').read_text())

    def test_initial_open_completion_precedes_the_first_phase_observation(self):
        source = make_paired_file_fixture(Path(self.temp.name) / 'initial-open', first_gripper_command=True)
        loaded = prepare.stream.load_source(source)
        self.assertGreater(prepare.inspect_source(loaded)['initial_hold_feedback_verified_frames'], 0)
        journal = prepare.stream.read_json(source / 'target-command-journal.json')
        initial_open = next(row for row in journal if row['teacher']['phase'] == 'initial_open')
        completed = prepare.stream.read_json(source / f"command-{initial_open['sequence']:04d}.json")[
            'completed_monotonic']
        path = source / 'jev-c01-approach_clear.json'
        record = prepare.stream.read_json(path)
        scene = record['state']['facts']['scene_observation']
        self.assertEqual(scene['previous_completion_monotonic'], completed)
        scene['previous_completion_monotonic'] = None
        record['request']['input'][0]['content'][0]['data'] = deepcopy(record['state'])
        fixtures.write_json(path, record)
        update_fixture_manifest(source, path)
        with self.assertRaisesRegex(ValueError, 'previous completed command'):
            prepare.inspect_source(prepare.stream.load_source(source))

    def test_initial_scene_and_snapshot_must_match_the_actual_pair(self):
        scene = self.record['state']['facts']['scene_observation']
        cases = [(('facts', 'scene_observation'), None),
                 (('facts', 'scene_observation', 'frame_index'), True),
                 (('facts', 'scene_observation', 'frame_index'), len(self.captures)),
                 (('facts', 'scene_observation', 'camera', 'source_camera_serial'), 'other-camera'),
                 (('facts', 'scene_observation', 'robot', 'tcp'), [0.] * 6),
                 (('facts', 'scene_observation', 'previous_completion_monotonic'), None),
                 (('captured_at_unix',), self.record['state']['captured_at_unix'] + 1),
                 (('camera', 'capture_time_monotonic'), scene['camera_time'] + 1),
                 (('robot', 'received_monotonic'), scene['robot_time_monotonic'] + 1),
                 (('robot', 'gripper_position'), 0),
                 (('robot', 'gripper_feedback_valid'), False),
                 (('facts', 'current_tcp_mm_deg'), [0.] * 6)]
        for path, value in cases:
            with self.subTest(field=path):
                record = deepcopy(self.record)
                target = record['state']
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.write_record(record)
                with self.assertRaisesRegex(ValueError, 'scene observation|paired capture frame|clock'):
                    self.inspect_fixture()

    def test_reusing_a_pair_from_before_the_previous_command_is_rejected(self):
        completed = self.record['state']['facts']['scene_observation']['previous_completion_monotonic']
        frame = next(frame for frame in self.captures
                     if frame['robot']['received_monotonic'] < completed)
        scene = {'frame_index': frame['frame_index'], 'camera': deepcopy(frame['camera']),
                 'robot': deepcopy(frame['robot']), 'clock': prepare.MONOTONIC,
                 'camera_time': frame['camera']['source_timestamp_monotonic'],
                 'robot_time_monotonic': frame['robot']['received_monotonic'],
                 'previous_completion_monotonic': completed}
        record = deepcopy(self.record)
        set_fixture_scene(record, scene)
        self.write_record(record)
        with self.assertRaisesRegex(ValueError, 'previous completed command'):
            self.inspect_fixture()

    def test_initial_decision_clocks_precede_the_first_command_for_the_phase(self):
        name = 'jev-c01-approach_clear.json'
        original = prepare.stream.read_json(self.source / name)
        issues = [row['issued_monotonic'] for row in self.journal if row['teacher']['decision_id'] == name]
        self.assertGreater(len(issues), 1)
        scene = original['state']['facts']['scene_observation']
        cases = [('request_started_monotonic', scene['camera_time'] - .01),
                 ('request_started_monotonic', scene['robot_time_monotonic'] - .001),
                 ('request_started_monotonic', None),
                 ('response_received_monotonic', original['request_started_monotonic'] - .01),
                 ('response_received_monotonic', (issues[0] + issues[1]) / 2),
                 ('response_received_monotonic', True)]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                record = deepcopy(original)
                record[field] = value
                self.write_record(record, name)
                with self.assertRaisesRegex(ValueError, 'clock'):
                    self.inspect_fixture()

    def test_stopped_phase_cannot_supply_executed_teacher_actions(self):
        for field, value in (('automatic_continuation', False), ('route', 'needs_llm'),
                             ('route', 'observe_scene'), ('observation_error', 'Fixture observation stopped')):
            with self.subTest(field=field, value=value):
                record = deepcopy(self.record)
                record[field] = value
                self.write_record(record)
                with self.assertRaisesRegex(ValueError, 'did not permit the executed teacher phase'):
                    self.inspect_fixture()


class OpenAIObserveSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='observe-redecision-file-fixture-')
        self.addCleanup(self.temp.cleanup)
        for target in ('fr5_jev.decision.call_jev', 'fr5_jev.decision.api_key'):
            guard = patch(target, side_effect=AssertionError('File fixtures cannot call an API or read credentials'))
            guard.start()
            self.addCleanup(guard.stop)
        self.source = make_observe_file_fixture(Path(self.temp.name) / 'source')
        self.base_name, self.name = 'jev-c01-release.json', 'jev-c01-release-after-observe.json'
        self.observation_name, self.image_name = 'observe-c01-release.json', 'observe-c01-release.jpg'
        self.base = prepare.stream.read_json(self.source / self.base_name)
        self.record = prepare.stream.read_json(self.source / self.name)
        self.observation = prepare.stream.read_json(self.source / self.observation_name)
        self.image_bytes = (self.source / self.image_name).read_bytes()
        self.manifest = prepare.stream.read_json(self.source / 'control-manifest.json')
        self.journal = prepare.stream.read_json(self.source / 'target-command-journal.json')

    def inspect_fixture(self):
        return prepare.inspect_source(prepare.stream.load_source(self.source))

    def write_record(self, name, record):
        path = self.source / name
        fixtures.write_json(path, record)
        update_fixture_manifest(self.source, path)

    def write_followup(self, record, observation=None):
        if observation is not None:
            record['state']['facts']['scene_observation'] = deepcopy(observation)
            self.write_record(self.observation_name, observation)
        record['request']['input'][0]['content'][0]['data'] = deepcopy(record['state'])
        self.write_record(self.name, record)

    def restore_chain(self):
        for name, record in ((self.base_name, self.base), (self.name, self.record),
                             (self.observation_name, self.observation)):
            fixtures.write_json(self.source / name, record)
        (self.source / self.image_name).write_bytes(self.image_bytes)
        fixtures.write_json(self.source / 'control-manifest.json', self.manifest)

    def test_observe_chain_is_accepted_without_changing_policy_rows(self):
        result = self.inspect_fixture()
        loaded = prepare.stream.load_source(self.source)
        clips, coverage = prepare.stream._build_clips(loaded, allow_initial_hold=True)
        self.assertEqual(result['coverage'], coverage)
        self.assertTrue(clips)
        self.assertGreater(result['initial_hold_feedback_verified_frames'], 0)
        self.assertEqual(self.base['answer']['choice'], 'observe_scene')
        self.assertEqual(self.record['answer']['choice'], 'release_gripper')
        self.assertEqual(len(self.base['request']['input'][0]['content']), 1)
        supplied = self.record['request']['input'][0]['content'][1]['image_url'].split(',', 1)[1]
        self.assertEqual(base64.b64decode(supplied), self.image_bytes)
        self.assertEqual(self.record['state']['facts']['scene_observation'], self.observation)
        self.assertEqual(loaded['review']['source'], 'direct_user_message')
        self.assertEqual({row['teacher']['decision_id'] for row in self.journal
                          if row['teacher']['phase'] == 'release'}, {self.name})
        self.assertIn('not a physical demonstration', (self.source / 'TEST_FIXTURE_ONLY').read_text())

    def test_base_response_and_final_phase_selection_are_required(self):
        for choice in ('release_gripper', 'hold', 'request_astra'):
            with self.subTest(base_choice=choice):
                base = deepcopy(self.base)
                set_fixture_choice(base, choice)
                self.write_record(self.base_name, base)
                with self.assertRaisesRegex(ValueError, 'base response did not request observe_scene'):
                    self.inspect_fixture()
        self.restore_chain()
        for choice in ('observe_scene', 'hold', 'request_astra'):
            with self.subTest(followup_choice=choice):
                record = deepcopy(self.record)
                set_fixture_choice(record, choice)
                self.write_followup(record)
                with self.assertRaisesRegex(ValueError, 'choice differs from the executed teacher phase'):
                    self.inspect_fixture()
        self.restore_chain()
        for name, original in ((self.base_name, self.base), (self.name, self.record)):
            with self.subTest(fabricated_answer=name):
                record = deepcopy(original)
                record['answer']['choice'] = 'hold'
                self.write_record(name, record)
                with self.assertRaisesRegex(ValueError, 'internal answer differs from its response'):
                    self.inspect_fixture()
                self.restore_chain()

    def test_both_decisions_require_raw_captured_model_evidence(self):
        for name, original in ((self.base_name, self.base), (self.name, self.record)):
            for field, value in (('request', None), ('response', None), ('state', None),
                                 ('selected_by', 'policy_evaluation'),
                                 ('decision_error', 'Test fixture inference failure')):
                with self.subTest(name=name, field=field):
                    self.restore_chain()
                    record = deepcopy(original)
                    record[field] = value
                    self.write_record(name, record)
                    with self.assertRaises(ValueError):
                        self.inspect_fixture()

    def test_executed_followup_filename_matches_its_phase_and_cycle(self):
        for name in ('jev-c02-release-after-observe.json', 'jev-c01-close-after-observe.json',
                     'jev-c01-release-after-observe-extra.json'):
            with self.subTest(name=name):
                self.write_record(name, self.record)
                journal = deepcopy(self.journal)
                for row in journal:
                    if row['teacher']['decision_id'] == self.name:
                        row['teacher']['decision_id'] = name
                write_fixture_journal(self.source, journal)
                with self.assertRaisesRegex(ValueError, 'filename differs from the teacher phase and cycle'):
                    self.inspect_fixture()

    def test_chain_links_filenames_and_manifest_evidence_are_required(self):
        cases = [(self.base_name, 'observation_record', 'observe-c02-release.json'),
                 (self.base_name, 'followup_decision_id', 'jev-c01-close-after-observe.json'),
                 (self.base_name, 'automatic_continuation', True),
                 (self.base_name, 'observation_error', 'Test fixture observation failure'),
                 (self.base_name, 'scope_sha256', 'other-fixture-scope'),
                 (self.name, 'automatic_continuation', False),
                 (self.observation_name, 'record', 'observe-c01-close.json'),
                 (self.observation_name, 'image_file', 'observe-c01-close.jpg'),
                 (self.observation_name, 'requested_by_decision', 'jev-c02-release.json')]
        for name, field, value in cases:
            with self.subTest(name=name, field=field):
                self.restore_chain()
                record = deepcopy({self.base_name: self.base, self.name: self.record,
                                   self.observation_name: self.observation}[name])
                record[field] = value
                if name == self.observation_name:
                    self.write_followup(deepcopy(self.record), record)
                else:
                    self.write_record(name, record)
                with self.assertRaises(ValueError):
                    self.inspect_fixture()
        for name in (self.base_name, self.name, self.observation_name, self.image_name):
            for case in ('missing_file', 'missing_manifest', 'changed_file'):
                with self.subTest(name=name, case=case):
                    self.restore_chain()
                    if case == 'missing_file':
                        (self.source / name).unlink()
                    elif case == 'missing_manifest':
                        manifest = deepcopy(self.manifest)
                        del manifest['files'][name]
                        fixtures.write_json(self.source / 'control-manifest.json', manifest)
                    else:
                        with (self.source / name).open('ab') as handle:
                            handle.write(b'\n')
                    with self.assertRaises(ValueError):
                        self.inspect_fixture()

    def test_request_requires_matching_json_and_inline_jpeg_bytes(self):
        content = self.record['request']['input'][0]['content']
        other = io.BytesIO()
        Image.new('RGB', (640, 360), (220, 10, 30)).save(other, format='JPEG')
        wrong_image = 'data:image/jpeg;base64,' + base64.b64encode(other.getvalue()).decode()
        invalid = [[content[0]], [content[0], {**content[1], 'image_url': wrong_image}],
                   [content[0], {**content[1], 'image_url': 'data:image/jpeg;base64,???'}],
                   [content[0], {**content[1], 'image_url': content[1]['image_url'].replace('jpeg', 'png')}],
                   [content[0], {**content[1], 'image_url': 'observe-c01-release.jpg'}],
                   [content[0], content[1], content[1]], [content[1], content[0]]]
        for value in invalid:
            with self.subTest(content=value[1].get('type') if len(value) > 1 else 'missing image'):
                record = deepcopy(self.record)
                record['request']['input'][0]['content'] = deepcopy(value)
                self.write_record(self.name, record)
                with self.assertRaises(ValueError):
                    self.inspect_fixture()
        record = deepcopy(self.record)
        record['request']['input'][0]['content'][0]['data']['facts']['current_tcp_mm_deg'][0] += 1
        self.write_record(self.name, record)
        with self.assertRaisesRegex(ValueError, 'request input differs from the recorded state'):
            self.inspect_fixture()

    def test_persisted_jpeg_must_be_the_observed_capture_image(self):
        for encoding in ('JPEG', 'PNG'):
            with self.subTest(encoding=encoding):
                image = io.BytesIO()
                Image.new('RGB', (640, 360), (250, 20, 30)).save(image, format=encoding)
                (self.source / self.image_name).write_bytes(image.getvalue())
                update_fixture_manifest(self.source, self.source / self.image_name)
                record = deepcopy(self.record)
                record['request']['input'][0]['content'][1]['image_url'] = (
                    'data:image/jpeg;base64,' + base64.b64encode(image.getvalue()).decode())
                self.write_followup(record)
                with self.assertRaisesRegex(ValueError, 'not a JPEG|differs from its captured frame image'):
                    self.inspect_fixture()

    def test_matching_inline_and_persisted_jpeg_cannot_reference_another_capture_frame(self):
        captures = prepare.stream.read_jsonl(self.source / 'continuous/capture.jsonl')
        base_frame = captures[self.base['state']['facts']['scene_observation']['frame_index']]
        other_bytes = (self.source / 'continuous' / base_frame['rgb_file']).read_bytes()
        self.assertNotEqual(other_bytes, self.image_bytes)
        (self.source / self.image_name).write_bytes(other_bytes)
        update_fixture_manifest(self.source, self.source / self.image_name)
        record = deepcopy(self.record)
        record['request']['input'][0]['content'][1]['image_url'] = (
            'data:image/jpeg;base64,' + base64.b64encode(other_bytes).decode())
        self.write_followup(record)
        with self.assertRaisesRegex(ValueError, 'differs from its captured frame image'):
            self.inspect_fixture()

    def test_observation_and_snapshot_match_the_actual_paired_frame(self):
        for field, value in (('frame_index', True), ('frame_index', 10000),
                             ('frame_index', self.observation['frame_index'] + 1),
                             ('clock', 'host_unix'), ('camera_time', self.observation['camera_time'] - .1),
                             ('robot_time_monotonic', self.observation['robot_time_monotonic'] - .1),
                             ('previous_completion_monotonic', None), ('synthetic', True)):
            with self.subTest(observation_field=field):
                observation = deepcopy(self.observation)
                observation[field] = value
                self.write_followup(deepcopy(self.record), observation)
                with self.assertRaises(ValueError):
                    self.inspect_fixture()
        for section, field, value in (('camera', 'source_camera_serial', 'other-camera'),
                                      ('robot', 'tcp', [0.] * 6),
                                      ('robot', 'received_monotonic', self.observation['camera_time'] - .1)):
            with self.subTest(observation_section=section, field=field):
                observation = deepcopy(self.observation)
                observation[section][field] = value
                self.write_followup(deepcopy(self.record), observation)
                with self.assertRaisesRegex(ValueError, 'differs from its paired capture frame'):
                    self.inspect_fixture()
        self.restore_chain()
        for section, field, value in (('camera', 'capture_time_monotonic', self.observation['camera_time'] - 1),
                                      ('camera', 'capture_time_unix', 2000),
                                      ('camera', 'serial', 'other-camera'),
                                      ('robot', 'gripper_position', 10),
                                      ('robot', 'gripper_motiondone', 0),
                                      ('robot', 'gripper_active', 0),
                                      ('robot', 'gripper_feedback_valid', False),
                                      ('robot', 'received_monotonic', self.observation['camera_time'] - 1),
                                      ('facts', 'current_tcp_mm_deg', [0.] * 6)):
            with self.subTest(snapshot_section=section, field=field):
                record = deepcopy(self.record)
                record['state'][section][field] = value
                self.write_followup(record)
                with self.assertRaisesRegex(ValueError, 'state differs from its captured scene observation'):
                    self.inspect_fixture()

    def test_new_observation_and_redecision_are_causally_ordered(self):
        cases = [(self.base_name, 'request_started_monotonic', self.base['response_received_monotonic'] + 1),
                 (self.base_name, 'response_received_monotonic', self.observation['camera_time'] + 1),
                 (self.base_name, 'response_received_monotonic', None),
                 (self.observation_name, 'requested_monotonic', self.base['request_started_monotonic']),
                 (self.observation_name, 'requested_monotonic', self.observation['camera_time']),
                 (self.observation_name, 'requested_monotonic', self.observation['robot_time_monotonic'] + 1),
                 (self.name, 'request_started_monotonic', self.observation['camera_time'] - .01),
                 (self.name, 'response_received_monotonic', self.record['request_started_monotonic'] - .01),
                 (self.name, 'response_received_monotonic', 1000),
                 (self.name, 'request_started_monotonic', True)]
        for name, field, value in cases:
            with self.subTest(name=name, field=field, value=value):
                self.restore_chain()
                record = deepcopy({self.base_name: self.base, self.name: self.record,
                                   self.observation_name: self.observation}[name])
                record[field] = value
                if name == self.observation_name:
                    self.write_followup(deepcopy(self.record), record)
                else:
                    self.write_record(name, record)
                with self.assertRaisesRegex(ValueError, 'clocks|clock'):
                    self.inspect_fixture()

    def test_image_evidence_does_not_replace_complete_direct_user_success(self):
        review_path = self.source / 'user-review.json'
        review = prepare.stream.read_json(review_path)
        for field, value in (('verdict', 'pending'), ('verdict', 'failure'),
                             ('reviewer', 'agent'), ('source', 'current_codex_chat'),
                             ('user_statement', '')):
            with self.subTest(review_field=field, value=value):
                fixtures.write_json(review_path, {**review, field: value})
                with self.assertRaisesRegex(ValueError, 'User success review'):
                    self.inspect_fixture()
        fixtures.write_json(review_path, review)
        summary_path = self.source / 'summary.json'
        summary = prepare.stream.read_json(summary_path)
        for field, value in (('completed_stages', ['return']), ('outcome', 'success'),
                             ('outcome', 'interrupted'), ('outcome', 'failed'),
                             ('failed', 'Test fixture failure')):
            with self.subTest(summary_field=field):
                fixtures.write_json(summary_path, {**summary, field: value})
                set_fixture_scope(self.source, 'training_complete_trial')
                with self.assertRaisesRegex(ValueError, 'Episode is incomplete'):
                    self.inspect_fixture()

    def test_observe_sources_still_reject_synthetic_and_policy_evaluation_data(self):
        scope_path = self.source / 'scope.json'
        scope = prepare.stream.read_json(scope_path)
        fixtures.write_json(scope_path, {**scope, 'synthetic': True})
        set_fixture_scope(self.source, 'training_complete_trial')
        with self.assertRaisesRegex(ValueError, 'captured fr5_jev'):
            self.inspect_fixture()
        fixtures.write_json(scope_path, scope)
        set_fixture_scope(self.source, 'policy_evaluation')
        with self.assertRaisesRegex(ValueError, 'complete-trial split'):
            self.inspect_fixture()
        set_fixture_scope(self.source, 'training_complete_trial')
        for field in ('policy', 'source'):
            with self.subTest(evaluation_field=field):
                journal = deepcopy(self.journal)
                if field == 'policy':
                    journal[0]['teacher']['policy'] = 'policy_evaluation'
                else:
                    journal[0]['source'] = 'policy_evaluation'
                write_fixture_journal(self.source, journal)
                with self.assertRaisesRegex(ValueError, 'fr5_jev teacher'):
                    self.inspect_fixture()
        write_fixture_journal(self.source, self.journal)
        captures_path = self.source / 'continuous/capture.jsonl'
        teachers_path = self.source / 'lerobot/meta/capture.jsonl'
        captures, teachers = prepare.stream.read_jsonl(captures_path), prepare.stream.read_jsonl(teachers_path)
        index = self.observation['frame_index']
        captures[index]['robot']['simulated'] = True
        teachers[index]['robot'] = captures[index]['robot']
        fixtures.write_jsonl(captures_path, captures)
        fixtures.write_jsonl(teachers_path, teachers)
        material_path = self.source / 'stream-materialization.json'
        material = prepare.stream.read_json(material_path)
        material['capture_journal_sha256'] = prepare.stream.sha(captures_path)
        material['teacher_journal_sha256'] = prepare.stream.sha(teachers_path)
        fixtures.write_json(material_path, material)
        with self.assertRaisesRegex(ValueError, 'Synthetic or simulated'):
            self.inspect_fixture()


class PreparationTests(unittest.TestCase):
    def test_existing_exporter_supports_a_train_only_dataset(self):
        with tempfile.TemporaryDirectory(prefix='jev-export-test-') as directory:
            root = Path(directory)
            source = fixtures.make_source(root / 'fixture')
            with patch.object(prepare.stream, 'DATA_ROOT', root):
                result = prepare.stream.export([source], root / 'output', validation_indices=[],
                                               allow_initial_hold=True)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['train_episode_indices'], [0])
            self.assertEqual(result['validation_episode_indices'], [])

    def test_outputs_keep_original_holdout_out_of_generic_training_dataset(self):
        with tempfile.TemporaryDirectory(prefix='jev-output-test-') as directory:
            root = Path(directory)
            sources = [{'trial': str(root / 'raw' / str(i))} for i in range(8)]
            report = {'sources': sources, 'train_source_indices': list(range(7)),
                      'holdout_source_indices': [7]}
            calls = []

            def export(paths, output, validation_indices, *, allow_initial_hold):
                calls.append(([str(p) for p in paths], output, validation_indices, allow_initial_hold))
                output.mkdir()
                return {'status': 'passed'}

            with patch.object(prepare, 'DATA_ROOT', root), patch.object(prepare, 'inspect', return_value=report), \
                    patch.object(prepare.stream, 'export', side_effect=export):
                result = prepare.prepare(root / 'new')
                self.assertEqual(calls[0][0], [s['trial'] for i, s in enumerate(sources) if i != 7])
                self.assertEqual(calls[0][2], [])
                self.assertEqual(calls[1][0], [s['trial'] for s in sources])
                self.assertEqual(calls[1][2], [7])
                self.assertTrue(all(call[3] for call in calls))
                self.assertEqual(json.loads((root / 'new/selection.json').read_text()), result)
                with self.assertRaises(FileExistsError):
                    prepare.prepare(root / 'new')


class StrictMaterializedIntegrationTests(unittest.TestCase):
    def test_real_inspection_and_export_preserve_source_splits_and_tasks(self):
        with tempfile.TemporaryDirectory(prefix='jev-strict-file-fixture-') as directory:
            root = Path(directory)
            originals = [make_jev_file_fixture(root / f'original-{index}', index,
                                               'training_complete_trial', '透明杯子')
                         for index in range(8)]
            train = make_observe_file_fixture(root / 'train-object', 8, 'training_complete_trial', '紫色方块')
            heldout = make_observe_file_fixture(root / 'heldout-object', 9, 'held_out_complete_trial', '绿色瓶子')
            reference = make_fixture_reference(root / 'reference', originals)
            output = root / 'output'
            with patch.object(prepare, 'DATA_ROOT', root), patch.object(prepare.stream, 'DATA_ROOT', root):
                result = prepare.prepare(output, reference=reference,
                                         train_sources=[train], validation_sources=[heldout])
                with self.assertRaises(FileExistsError):
                    prepare.prepare(output, reference=reference)
            self.assertEqual(result['train_source_indices'], list(range(7)) + [8])
            self.assertEqual(result['holdout_source_indices'], [7, 9])
            self.assertEqual(result['sources'][8]['object_description'], '紫色方块')
            self.assertEqual(result['sources'][9]['object_description'], '绿色瓶子')
            self.assertEqual(result['sources'][9]['scope_split_role'], 'held_out_complete_trial')
            self.assertEqual(result['sources'][9]['user_verdict'], 'success')
            self.assertGreater(result['sources'][9]['initial_hold_feedback_verified_frames'], 0)

            train_proof = json.loads((output / 'train/meta/stream_provenance.json').read_text())
            combined_proof = json.loads((output / 'with-validation/meta/stream_provenance.json').read_text())
            combined_stats = json.loads((output / 'with-validation/meta/stream_stats.json').read_text())
            self.assertEqual([Path(item['trial']) for item in train_proof['sources']], originals[:7] + [train])
            self.assertEqual([Path(item['trial']) for item in combined_proof['sources']],
                             originals + [train, heldout])
            self.assertEqual([item['split'] for item in combined_proof['sources']],
                             ['train'] * 7 + ['validation', 'train', 'validation'])
            self.assertEqual(combined_proof['sources'][9]['task_instruction'],
                             '抓取桌面上的绿色瓶子，在空中维持5秒后放下')
            self.assertEqual(combined_proof['sources'][9]['task_spec']['object_description'], '绿色瓶子')
            self.assertEqual(combined_stats['source_indices'], list(range(7)) + [8])
            self.assertEqual(combined_stats['episode_indices'], combined_proof['train_episode_indices'])
            self.assertEqual(combined_proof['validation_episode_indices'], [7, 9])
            import numpy as np
            import pyarrow.parquet as pq
            rows = pq.read_table(sorted((output / 'with-validation/data').rglob('*.parquet'))).to_pylist()
            training = np.asarray([row['observation.state'] for row in rows
                                   if row['episode_index'] in combined_proof['train_episode_indices']])
            all_states = np.asarray([row['observation.state'] for row in rows])
            mean = combined_stats['stats']['observation.state']['mean']
            np.testing.assert_allclose(mean, training.mean(axis=0), rtol=1e-6, atol=1e-7)
            self.assertFalse(np.allclose(mean, all_states.mean(axis=0)))
            self.assertEqual(json.loads((output / 'selection.json').read_text()), result)
            with self.assertRaisesRegex(ValueError, 'training_complete_trial'):
                prepare.inspect(reference, train_sources=[heldout])

    def test_actual_heldout_source_rejects_missing_jev_feedback_and_review(self):
        with tempfile.TemporaryDirectory(prefix='jev-strict-rejection-fixture-') as directory:
            root = Path(directory)
            source = make_jev_file_fixture(root / 'heldout', 9, 'held_out_complete_trial', '绿色瓶子')
            loaded = prepare.stream.load_source(source)
            self.assertGreater(prepare.inspect_source(loaded)['initial_hold_feedback_verified_frames'], 0)

            decision_path = source / 'jev-c01-close.json'
            decision = json.loads(decision_path.read_text())
            fixtures.write_json(decision_path, {key: value for key, value in decision.items() if key != 'selected_by'})
            update_fixture_manifest(source, decision_path)
            with self.assertRaisesRegex(ValueError, 'phase decision does not match'):
                prepare.inspect_source(prepare.stream.load_source(source))
            fixtures.write_json(decision_path, decision)
            update_fixture_manifest(source, decision_path)

            review_path = source / 'user-review.json'
            review = json.loads(review_path.read_text())
            for verdict in ('pending', 'failure'):
                with self.subTest(verdict=verdict):
                    fixtures.write_json(review_path, {**review, 'verdict': verdict})
                    with self.assertRaisesRegex(ValueError, 'User success review'):
                        prepare.stream.load_source(source)
            fixtures.write_json(review_path, review)

            captures_path = source / 'continuous/capture.jsonl'
            teachers_path = source / 'lerobot/meta/capture.jsonl'
            captures = prepare.stream.read_jsonl(captures_path)
            teachers = prepare.stream.read_jsonl(teachers_path)
            for capture, teacher in zip(captures[:10], teachers[:10]):
                capture['robot']['gripper_position'] = 82
                teacher['robot'] = capture['robot']
            fixtures.write_jsonl(captures_path, captures)
            fixtures.write_jsonl(teachers_path, teachers)
            material_path = source / 'stream-materialization.json'
            material = json.loads(material_path.read_text())
            material['capture_journal_sha256'] = prepare.stream.sha(captures_path)
            material['teacher_journal_sha256'] = prepare.stream.sha(teachers_path)
            fixtures.write_json(material_path, material)
            with self.assertRaisesRegex(ValueError, 'valid recorded feedback'):
                prepare.inspect_source(prepare.stream.load_source(source))


class MaterializedSourceIntegrationTests(unittest.TestCase):
    def make_object_source(self, root, trial, role, description):
        instruction = f'抓取桌面上的{description}，在空中维持5秒后放下'
        source = fixtures.make_source(root, trial=trial, instruction=instruction)
        return set_fixture_scope(source, role, object_description=description)

    def test_inspection_loads_actual_materialized_sources_and_rejects_synthetic(self):
        with tempfile.TemporaryDirectory(prefix='jev-materialized-test-') as directory:
            root = Path(directory)
            originals, records = [], []
            for index in range(8):
                path = fixtures.make_source(root / f'original-{index}', trial=index)
                source = prepare.stream.load_source(path)
                originals.append(path)
                records.append({'trial': str(source['root']),
                                'source_scope_sha256': source['source_scope_sha256'],
                                'source_summary_sha256': source['source_summary_sha256'],
                                'user_review_sha256': source['review_sha256'],
                                'session': source['materialization']['session'],
                                'source_stream_sha256': source['materialization']['source_stream_sha256']})
            train = self.make_object_source(root / 'train', 8, 'training_complete_trial', '紫色方块')
            heldout = self.make_object_source(root / 'heldout', 9, 'held_out_complete_trial', '绿色瓶子')

            def fixture_report(source):
                return {'trial': str(source['root']), 'task_instruction': source['task_instruction'],
                        'user_verdict': source['review']['verdict'], 'coverage': {'exported_frames': 1}}

            actual_inspect = prepare.inspect_source
            with patch.object(prepare, 'reference_sources', return_value=records), \
                 patch.object(prepare, 'inspect_source', side_effect=fixture_report):
                report = prepare.inspect(root / 'reference', train_sources=[train], validation_sources=[heldout])
            self.assertEqual([Path(item['trial']) for item in report['sources']], originals + [train, heldout])
            self.assertEqual(report['train_source_indices'], list(range(7)) + [8])
            self.assertEqual(report['holdout_source_indices'], [7, 9])
            self.assertEqual(report['sources'][8]['object_description'], '紫色方块')
            self.assertEqual(report['sources'][9]['object_description'], '绿色瓶子')
            self.assertEqual(report['sources'][9]['scope_split_role'], 'held_out_complete_trial')
            self.assertEqual(report['sources'][9]['task_instruction'],
                             '抓取桌面上的绿色瓶子，在空中维持5秒后放下')

            def inspect_new_sources(source):
                return fixture_report(source) if source['root'] in originals else actual_inspect(source)

            with patch.object(prepare, 'reference_sources', return_value=records), \
                 patch.object(prepare, 'inspect_source', side_effect=inspect_new_sources):
                for options in ({'train_sources': [train]}, {'validation_sources': [heldout]}):
                    with self.subTest(options=options):
                        with self.assertRaisesRegex(ValueError, 'captured fr5_jev'):
                            prepare.inspect(root / 'reference', **options)

    def test_actual_exporter_keeps_held_out_object_and_train_only_stats(self):
        with tempfile.TemporaryDirectory(prefix='jev-heldout-export-test-') as directory:
            root = Path(directory)
            train = self.make_object_source(root / 'train', 0, 'training_complete_trial', '紫色方块')
            heldout = self.make_object_source(root / 'heldout', 1, 'held_out_complete_trial', '绿色瓶子')
            output = root / 'dataset'
            with patch.object(prepare.stream, 'DATA_ROOT', root):
                result = prepare.stream.export([train, heldout], output, validation_indices=[1],
                                               allow_initial_hold=True)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['train_episode_indices'], [0])
            self.assertEqual(result['validation_episode_indices'], [1])
            proof = json.loads((output / 'meta/stream_provenance.json').read_text())
            stats = json.loads((output / 'meta/stream_stats.json').read_text())
            self.assertEqual([source['split'] for source in proof['sources']], ['train', 'validation'])
            self.assertEqual(proof['sources'][1]['task_spec']['object_description'], '绿色瓶子')
            self.assertEqual(proof['sources'][1]['task_instruction'],
                             '抓取桌面上的绿色瓶子，在空中维持5秒后放下')
            self.assertEqual(proof['training_data_kind'], 'synthetic_diagnostic')
            self.assertEqual(stats['source_indices'], [0])
            self.assertEqual(stats['stats']['observation.state']['count'], [result['train_frames']])


class AdditionalSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='jev-additional-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / 'reference'
        self.sources = {}
        self.records = []
        for index in range(8):
            source = self.add_source(f'original-{index}', 'training_complete_trial', '透明杯子')
            materialization = source['materialization']
            self.records.append({'trial': str(source['root']),
                                 'source_scope_sha256': source['source_scope_sha256'],
                                 'source_summary_sha256': source['source_summary_sha256'],
                                 'user_review_sha256': source['review_sha256'],
                                 'session': materialization['session'],
                                 'source_stream_sha256': materialization['source_stream_sha256']})
        self.real_load_source = prepare.stream.load_source
        self.real_inspect_source = prepare.inspect_source
        for target, kwargs in ((patch.object(prepare, 'reference_sources'), {'return_value': self.records}),
                               (patch.object(prepare.stream, 'load_source'), {'side_effect': self.load_source}),
                               (patch.object(prepare, 'inspect_source'), {'side_effect': self.inspect_source})):
            mock = target.start()
            mock.configure_mock(**kwargs)
            self.addCleanup(target.stop)

    def add_source(self, name, role, description):
        path = self.root / 'recordings' / name
        task = f'抓取桌面上的{description}，在空中维持5秒后放下'
        source = {'root': path, 'scope': {'split_role': role},
                  'task_instruction': task, 'task_spec': {'object_description': description},
                  'source_scope_sha256': f'scope-{name}', 'source_summary_sha256': f'summary-{name}',
                  'review_sha256': f'review-{name}', 'review': {'verdict': 'success'},
                  'materialization': {'session': f'session-{name}', 'source_stream_sha256': f'stream-{name}'}}
        self.sources[path] = source
        return source

    def load_source(self, path):
        path = Path(path)
        if path in self.sources:
            return self.sources[path]
        return self.real_load_source(path)

    def inspect_source(self, source):
        if source['root'] not in self.sources:
            return self.real_inspect_source(source)
        return {'trial': str(source['root']), 'session': source['materialization']['session'],
                'user_verdict': source['review']['verdict'], 'task_instruction': source['task_instruction'],
                'coverage': {'exported_frames': 10}}

    def test_no_additions_keep_the_original_source_split(self):
        report = prepare.inspect(self.reference)
        self.assertEqual([item['trial'] for item in report['sources']],
                         [record['trial'] for record in self.records])
        self.assertEqual(report['train_source_indices'], list(range(7)))
        self.assertEqual(report['holdout_source_indices'], [7])

    def test_additional_roles_order_language_and_export_splits(self):
        train = [self.add_source('train-block', 'training_complete_trial', '紫色方块'),
                 self.add_source('train-bottle', 'training_complete_trial', '绿色瓶子')]
        validation = [self.add_source('val-block', 'held_out_complete_trial', '蓝色方块'),
                      self.add_source('val-bottle', 'held_out_complete_trial', '红色瓶子')]
        train_paths = [source['root'] for source in train]
        validation_paths = [source['root'] for source in validation]
        calls = []

        def export(paths, output, validation_indices, *, allow_initial_hold):
            calls.append((list(paths), output, list(validation_indices), allow_initial_hold))
            output.mkdir()
            return {'status': 'passed'}

        with patch.object(prepare, 'DATA_ROOT', self.root), \
             patch.object(prepare.stream, 'export', side_effect=export):
            report = prepare.prepare(self.root / 'output', reference=self.reference,
                                     train_sources=train_paths, validation_sources=validation_paths)
        originals = [Path(record['trial']) for record in self.records]
        all_paths = originals + train_paths + validation_paths
        self.assertEqual([Path(item['trial']) for item in report['sources']], all_paths)
        self.assertEqual(report['train_source_indices'], list(range(7)) + [8, 9])
        self.assertEqual(report['holdout_source_indices'], [7, 10, 11])
        self.assertEqual(calls[0][0], originals[:7] + train_paths)
        self.assertEqual(calls[0][2], [])
        self.assertEqual(calls[1][0], all_paths)
        self.assertEqual(calls[1][2], [7, 10, 11])
        self.assertTrue(all(call[3] for call in calls))
        self.assertEqual(report['sources'][7]['split'], 'validation')
        self.assertEqual(report['sources'][7]['scope_split_role'], 'training_complete_trial')
        self.assertEqual(report['sources'][8]['split'], 'train')
        self.assertEqual(report['sources'][8]['object_description'], '紫色方块')
        self.assertEqual(report['sources'][8]['task_instruction'], train[0]['task_instruction'])
        self.assertEqual(report['sources'][10]['split'], 'validation')
        self.assertEqual(report['sources'][10]['scope_split_role'], 'held_out_complete_trial')
        self.assertEqual(report['sources'][10]['object_description'], '蓝色方块')
        self.assertEqual(report['sources'][10]['task_instruction'], validation[0]['task_instruction'])
        self.assertEqual(json.loads((self.root / 'output/selection.json').read_text()), report)

    def test_duplicate_paths_are_rejected_before_loading(self):
        extra = self.root / 'new-source'
        original = Path(self.records[2]['trial'])
        for train, validation in (([extra, extra], []), ([extra], [extra]), ([original], []),
                                  ([], [original])):
            with self.subTest(train=train, validation=validation):
                prepare.stream.load_source.reset_mock()
                with self.assertRaisesRegex(ValueError, 'Duplicate source path'):
                    prepare.inspect(self.reference, train_sources=train, validation_sources=validation)
                prepare.stream.load_source.assert_not_called()

    def test_additional_source_role_must_match_requested_split(self):
        training = self.add_source('wrong-train', 'held_out_complete_trial', '方块')
        heldout = self.add_source('wrong-validation', 'training_complete_trial', '方块')
        with self.assertRaisesRegex(ValueError, 'training_complete_trial'):
            prepare.inspect(self.reference, train_sources=[training['root']])
        with self.assertRaisesRegex(ValueError, 'held_out_complete_trial'):
            prepare.inspect(self.reference, validation_sources=[heldout['root']])

    def test_output_cannot_be_created_inside_reference_dataset(self):
        output = self.reference / 'new-output'
        with patch.object(prepare, 'DATA_ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'reference dataset directories must not overlap'):
                prepare.prepare(output, reference=self.reference)
        self.assertFalse(output.exists())
        prepare.stream.load_source.assert_not_called()

    def test_real_source_loader_rejects_failure_pending_and_synthetic_fixture(self):
        source = fixtures.make_source(self.root / 'synthetic', trial=20)
        review_path = source / 'user-review.json'
        review = json.loads(review_path.read_text())
        for verdict in ('failure', 'pending'):
            with self.subTest(verdict=verdict):
                review_path.write_text(json.dumps({**review, 'verdict': verdict}))
                with self.assertRaisesRegex(ValueError, 'User success review'):
                    prepare.inspect(self.reference, train_sources=[source])
        review_path.unlink()
        with self.assertRaises(FileNotFoundError):
            prepare.inspect(self.reference, train_sources=[source])
        review_path.write_text(json.dumps(review))
        original_read = whole_task_source.read_json

        def read_failed_summary(path):
            value = original_read(path)
            if Path(path) == source / 'summary.json':
                return {**value, 'outcome': 'failed', 'failed': 'fixture fault'}
            return value

        with patch.object(whole_task_source, 'read_json', side_effect=read_failed_summary):
            with self.assertRaisesRegex(ValueError, 'Episode is incomplete'):
                prepare.inspect(self.reference, train_sources=[source])
        with self.assertRaisesRegex(ValueError, 'captured fr5_jev'):
            prepare.inspect(self.reference, train_sources=[source])

    def test_cli_accepts_repeated_sources_for_inspection_and_export(self):
        options = ['--train-source', '/train-a', '--train-source', '/train-b',
                   '--validation-source', '/validation-a', '--validation-source', '/validation-b']
        expected = {'train_sources': [Path('/train-a'), Path('/train-b')],
                    'validation_sources': [Path('/validation-a'), Path('/validation-b')]}
        with patch.object(prepare, 'inspect', return_value={'status': 'inspected'}) as inspect, \
             patch('sys.argv', ['prepare_jev_pi05_dataset', *options]), redirect_stdout(io.StringIO()) as output:
            prepare.main()
        inspect.assert_called_once_with(**expected)
        self.assertEqual(json.loads(output.getvalue())['status'], 'inspected')
        with patch.object(prepare, 'prepare', return_value={'status': 'prepared'}) as export, \
             patch('sys.argv', ['prepare_jev_pi05_dataset', '--output', '/new', *options]), \
             redirect_stdout(io.StringIO()) as output:
            prepare.main()
        export.assert_called_once_with(Path('/new'), **expected)
        self.assertEqual(json.loads(output.getvalue())['status'], 'prepared')


if __name__ == '__main__':
    unittest.main()
