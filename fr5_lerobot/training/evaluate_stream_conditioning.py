"""Compare one sealed FR5 stream checkpoint using offline observations only.

Run each checkpoint in its own process. This program never imports a robot or
camera driver, updates weights, or writes into a checkpoint or dataset.
"""
from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


sys.dont_write_bytecode = True
ROOT = Path('/home/data4/hts/demo1/fr5-learning')
AUDIT_ROOT = ROOT / 'audits/pi05-conditioned-comparison-20261103-v1'
TRAINING_LOG = ROOT / 'audits/pi05-conditioned-formal-20261103-v1/training.log'
TRIAL = ROOT / 'recording-spool/pi05-workspace-only-trial-20261102-08/materialized'
HOVER_EVIDENCE = ROOT / 'audits/fr5-hover-observation-ambiguity-20261103-v1/evidence.json'
PARENT_SHA = 'f507a414d77fa5d957742d05a11aef048514306aa71e56b8944d3718fca8a516'
DATASET_SHA = '975ff43702e29ae5b1dcf2467a363452493d86b4839dade4e12dbd1500cfcc24'
BASE_SEED = 20261103
SEED_STRIDE = 1000
HISTORY_OFFSETS_S = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
OPEN = 1.0
CLOSE = 0.76


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode())


def seeds_for_case(case_index, draws, base_seed=BASE_SEED):
    if not 0 <= case_index or not 1 <= draws <= SEED_STRIDE:
        raise ValueError('Invalid case index or draw count')
    return [base_seed + case_index * SEED_STRIDE + draw for draw in range(draws)]


def select_known_cases(rows, episodes, events, trial15_state):
    """Select real starts, command transitions, and one valid approach per source."""
    if len(trial15_state) != 8:
        raise ValueError('Historical request must have an eight-element state')
    first_close = {}
    for event in events['close']:
        ep, pos = event['episode'], event['position']
        if ep not in episodes or not 0 <= pos < len(episodes[ep]) or episodes[ep][pos] != event['index']:
            raise ValueError('Close transition does not agree with episode')
        if ep not in first_close or pos < first_close[ep]:
            first_close[ep] = pos
    by_source = {}
    offsets = {}
    for ep, indices in episodes.items():
        for position, index in enumerate(indices):
            if index in offsets:
                raise ValueError('Overlapping episodes')
            offsets[index] = (ep, position)
            source = str(rows[index]['_physical_source'])
            by_source.setdefault(source, [])
            state = rows[index]['observation.state']
            if (ep not in first_close or position >= first_close[ep] or len(state) != 8
                    or float(state[7]) != 1.0 or not .98 <= float(state[6]) <= 1.0):
                continue
            future = indices[position:position + 50]
            if len(future) != 50 or any(float(rows[i]['action'][6]) != OPEN for i in future):
                continue
            distance = math.sqrt(sum((float(state[j]) - float(trial15_state[j])) ** 2 for j in range(6)))
            by_source[source].append((distance, index, ep, position))
    result = []
    for ep, indices in episodes.items():
        if not indices:
            raise ValueError('Empty episode')
        result.append({'kind': 'source_start', 'episode': ep, 'position': 0, 'row_index': indices[0]})
    for kind in ('close', 'open'):
        for event_number, event in enumerate(events[kind]):
            ep, pos, index = event['episode'], event['position'], event['index']
            if ep not in episodes or not 0 <= pos < len(episodes[ep]) or episodes[ep][pos] != index:
                raise ValueError('Transition does not agree with episode')
            result.append({'kind': kind + '_transition', 'episode': ep, 'position': pos,
                           'row_index': index, 'event_number': event_number,
                           'command_sequence': event['command_sequence'],
                           'previous_gripper_target': float(event['old']),
                           'new_gripper_target': float(event['new'])})
    for source, candidates in by_source.items():
        if not candidates:
            raise ValueError(f'Source {source} has no valid 50-step approach-open anchor')
        distance, index, ep, position = min(candidates, key=lambda x: (x[0], x[1]))
        result.append({'kind': 'approach_open', 'episode': ep, 'position': position,
                       'row_index': index, 'eligible_count': len(candidates),
                       'first_close_row_index': episodes[ep][first_close[ep]],
                       'distance_to_trial08_chunk15_q_l2_rad': distance})
    for case in result:
        row = rows[case['row_index']]
        case['source_index'] = int(row['_physical_source'])
        case['split'] = row['_split']
        if case['split'] not in ('train', 'validation'):
            raise ValueError('Unknown source split')
    result.sort(key=lambda x: (x['source_index'], x['row_index'], x['kind']))
    for case_index, case in enumerate(result):
        case['case_index'] = case_index
    return result


def select_hover_cases(rows, episodes, proof, evidence):
    """Select the first and last observed rows in each recorded hold."""
    pairs = ((0, 291, 321), (1, 821, 840), (7, 3374, 3398))
    examples = {item['source']: item for item in evidence['examples']}
    sources = {int(item['source_index']): item for item in proof['sources']}
    source_episodes = {}
    for ep, indices in episodes.items():
        source = int(rows[indices[0]]['_physical_source'])
        source_episodes.setdefault(source, []).append((ep, indices))
    holds = {}
    for source, source_info in sorted(sources.items()):
        for hold_index, hold in enumerate(source_info['hold_records']):
            start, end = hold['start_monotonic'], hold['end_monotonic']
            matches = []
            for ep, indices in source_episodes.get(source, []):
                inside = [(pos, index) for pos, index in enumerate(indices)
                          if int(rows[index]['_physical_source']) == source
                          and start <= rows[index]['_grid_time'] <= end]
                if inside:
                    matches.append((ep, inside))
            if len(matches) != 1 or len(matches[0][1]) < 2:
                raise ValueError('Recorded hold must match endpoints in one episode')
            ep, inside = matches[0]
            selected = []
            for label, (pos, index) in (('hold_start', inside[0]), ('hold_end', inside[-1])):
                row = rows[index]
                if row['_split'] != source_info['split'] or len(episodes[ep][pos:pos + 10]) != 10:
                    raise ValueError('Hover anchor does not agree with its recorded hold')
                selected.append({'kind': label, 'source_index': source, 'hold_index': hold_index,
                                 'repetition': hold['repetition'], 'split': row['_split'],
                                 'episode': ep, 'position': pos, 'row_index': index,
                                 'hold_elapsed_s': row['_grid_time'] - start,
                                 'hold_remaining_s': end - row['_grid_time']})
            holds[(source, hold_index)] = selected
    result = []
    legacy_holds = set()
    for source, first, last in pairs:
        example = examples[source]
        if (example['early']['row'], example['late']['row']) != (first, last):
            raise ValueError('Hover audit rows differ from the requested anchors')
        source_info = sources[source]
        hold_index = next(i for i, h in enumerate(source_info['hold_records'])
                          if h['repetition'] == example['repetition'])
        key = (source, hold_index)
        selected = holds[key]
        if tuple(case['row_index'] for case in selected) != (first, last):
            raise ValueError('Hover anchor does not agree with its recorded hold')
        for case, audit in zip(selected, (example['early'], example['late'])):
            if (case['split'] != example['split']
                    or abs(case['hold_elapsed_s'] - audit['elapsed_s']) > 1e-5):
                raise ValueError('Hover anchor does not agree with its recorded hold')
            case.update(recorded_frame_index=audit['frame'], hold_elapsed_s=audit['elapsed_s'],
                        hold_remaining_s=audit['remaining_s'], evidence_task=example['task'])
        result.extend(selected)
        legacy_holds.add(key)
    for key, selected in holds.items():
        if key not in legacy_holds:
            result.extend(selected)
    for case_index, case in enumerate(result, start=38):
        case['case_index'] = case_index
    return result


def hover_joint_metrics(actions, targets, state, return_direction):
    """Compare ten joint targets and project motion onto the recorded return direction."""
    if len(actions) != 10 or len(targets) != 10 or len(state) != 8 or len(return_direction) != 6:
        raise ValueError('Expected a six-joint ten-step hover diagnostic')
    scale = 180 / math.pi
    errors = [[(float(a[j]) - float(t[j])) * scale for j in range(6)] for a, t in zip(actions, targets)]
    pred = [(float(actions[-1][j]) - float(state[j])) * scale for j in range(6)]
    target = [(float(targets[-1][j]) - float(state[j])) * scale for j in range(6)]
    return {'joint_rmse_deg': math.sqrt(sum(x*x for row in errors for x in row) / 60),
            'joint_rmse_deg_per_joint': [math.sqrt(sum(row[j]**2 for row in errors) / 10) for j in range(6)],
            'predicted_step10_delta_deg': pred, 'target_step10_delta_deg': target,
            'predicted_step10_return_projection_deg': sum(x*y for x, y in zip(pred, return_direction)),
            'target_step10_return_projection_deg': sum(x*y for x, y in zip(target, return_direction)),
            'predicted_first10_max_abs_delta_deg': max(abs((float(a[j])-float(state[j])) * scale)
                                                        for a in actions for j in range(6)),
            'target_first10_max_abs_delta_deg': max(abs((float(a[j])-float(state[j])) * scale)
                                                     for a in targets for j in range(6))}


def summarize_hover_pairs(cases):
    result = []
    holds = {}
    for case in cases:
        key = (case['source_index'], case['hold_index'])
        holds.setdefault(key, {})[case['kind']] = case
    for (source, hold_index), pair in holds.items():
        early, late = pair['hold_start'], pair['hold_end']
        early_values = [d['metrics']['predicted_step10_return_projection_deg'] for d in early['draws']]
        late_values = [d['metrics']['predicted_step10_return_projection_deg'] for d in late['draws']]
        result.append({'source_index': source, 'hold_index': hold_index, 'repetition': early['repetition'],
                       'split': early['split'],
                       'hold_start_row': early['row_index'], 'hold_end_row': late['row_index'],
                       'target_step10_return_projection_deg': [early['draws'][0]['metrics']['target_step10_return_projection_deg'],
                                                               late['draws'][0]['metrics']['target_step10_return_projection_deg']],
                       'prediction_mean_projection_deg': [sum(early_values)/len(early_values),
                                                          sum(late_values)/len(late_values)],
                       'prediction_mean_end_minus_start_deg': sum(late_values)/len(late_values) - sum(early_values)/len(early_values)})
    return result


def grip_metrics(actions, targets):
    """Recorded-target agreement, not an estimate of whole-task success."""
    if len(actions) != 10 or len(targets) != 10:
        raise ValueError('Expected ten action targets')
    matches = []
    errors = []
    for action, target in zip(actions, targets):
        predicted, truth = float(action[6]), float(target[6])
        if abs(truth - OPEN) < 1e-5:
            other = CLOSE
        elif abs(truth - CLOSE) < 1e-5:
            other = OPEN
        else:
            raise ValueError(f'Unexpected demonstrated gripper target {truth}')
        errors.append(abs(predicted - truth))
        matches.append(abs(predicted - truth) < abs(predicted - other))
    joint_errors = [abs(float(action[j]) - float(target[j])) * 180 / math.pi
                    for action, target in zip(actions, targets) for j in range(6)]
    return {'gripper_first10_mae_fraction': sum(errors) / 10,
            'matching_first10_steps': sum(matches), 'first10_all_match': all(matches),
            'first_step_matches': matches[0],
            'joint_first10_mae_deg_diagnostic': sum(joint_errors) / len(joint_errors)}


def aggregate(cases):
    groups = {}
    for split in ('train', 'validation'):
        groups[split] = {}
        for kind in ('source_start', 'approach_open', 'close_transition', 'open_transition'):
            anchors = [case for case in cases if case['split'] == split and case['kind'] == kind]
            if not anchors:
                continue
            draws = [draw for anchor in anchors for draw in anchor['draws']]
            groups[split][kind] = {
                'anchors': len(anchors), 'draws': len(draws),
                'matching_steps': sum(x['metrics']['matching_first10_steps'] for x in draws),
                'total_steps': len(draws) * 10,
                'draws_all_first10_match': sum(x['metrics']['first10_all_match'] for x in draws),
                'gripper_first10_mae_fraction_mean': sum(x['metrics']['gripper_first10_mae_fraction'] for x in draws) / len(draws),
                'joint_first10_mae_deg_mean_diagnostic': sum(x['metrics']['joint_first10_mae_deg_diagnostic'] for x in draws) / len(draws),
                'per_anchor_all_first10_match_counts': [
                    {'case_index': a['case_index'], 'row_index': a['row_index'],
                     'matching_draws': sum(d['metrics']['first10_all_match'] for d in a['draws']),
                     'draws': len(a['draws'])} for a in anchors]}
    return groups


def gpu_snapshot(index):
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    proc = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                           '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    rows = [line.split(', ') for line in gpu.stdout.strip().splitlines()]
    row = next((row for row in rows if int(row[0]) == index), None)
    if row is None:
        raise RuntimeError(f'GPU{index} was not found; refusing model load')
    processes = [line.split(', ') for line in proc.stdout.strip().splitlines() if line.strip()]
    return {'index': int(row[0]), 'name': row[1], 'uuid': row[2], 'memory_used_mib': int(row[3]),
            'memory_total_mib': int(row[4]), 'utilization_percent': int(row[5]),
            'compute_processes': [p for p in processes if p[0] == row[2]]}


def require_free_gpu(snapshot, expected_index=8):
    if (expected_index < 0 or snapshot['index'] != expected_index or snapshot['compute_processes'] or snapshot['utilization_percent'] > 5
            or snapshot['memory_total_mib'] - snapshot['memory_used_mib'] < 22000):
        raise RuntimeError(f'GPU{expected_index} is not sufficiently idle or has the wrong index; refusing model load')


def require_first_gradient(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get('status') == 'first_gradient':
                return item
    raise RuntimeError('Formal training has not logged first_gradient; refusing model load')


def reconstruct_capture_history(captures, request, frame_index):
    """Reconstruct causal history from the recorded camera/state pairs."""
    history, provenance = [None] * 6, [None] * 6
    if request.get('source_clock') != 'host_monotonic':
        return history, provenance
    now = request['source_time']
    eligible = []
    for capture in captures:
        if capture['frame_index'] > frame_index:
            continue
        stamp = capture.get('camera', {}).get('source_timestamp_monotonic')
        if type(stamp) not in (int, float) or not math.isfinite(stamp):
            return history, provenance
        if stamp <= now:
            eligible.append((stamp, capture))
    eligible = eligible[-128:]
    if any(a[0] > b[0] for a, b in zip(eligible, eligible[1:])):
        raise ValueError('Recorded capture history is not in camera time order')
    stamps = [item[0] for item in eligible]
    for slot, offset in enumerate(HISTORY_OFFSETS_S):
        index = bisect_right(stamps, now - offset) - 1
        if index < 0:
            continue
        stamp, capture = eligible[index]
        robot = capture.get('robot')
        valid = capture.get('gripper_feedback_valid')
        if not isinstance(robot, dict) or type(valid) is not bool:
            continue
        joints, received, position = robot.get('joints'), robot.get('received_monotonic'), robot.get('gripper_position')
        if (not isinstance(joints, (list, tuple)) or len(joints) != 6
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in joints)
                or type(received) not in (int, float) or not math.isfinite(received) or received > now
                or (valid and (type(position) not in (int, float) or not math.isfinite(position)
                               or not 0 <= position <= 100))):
            continue
        state = [math.radians(value) for value in joints] + [position / 100 if valid else 0., float(valid)]
        history[slot] = {'age_s': now - stamp, 'state': state}
        provenance[slot] = {'frame_index': capture['frame_index'], 'camera_time_monotonic': stamp,
                            'robot_received_monotonic': received}
    return history, provenance


def known_case_batch(trainer, model, index, rows, episodes, dataset, profile):
    kwargs = {'labels': False, 'image_profile': model.info['image_profile']}
    if profile != 'none':
        kwargs['observation_context_profile'] = profile
    return trainer.batch_for([index], rows, episodes, dataset, model.pre, **kwargs)


def load_trial(trial=TRIAL):
    """Authenticate recorded requests and their exact captured JPEGs, never use as labels."""
    trial = Path(trial)
    index_file = trial / 'stream-prediction/index.json'
    index = json.loads(index_file.read_text(encoding='utf-8'))
    records = []
    for part in index['chunks']:
        path = (index_file.parent / part['path']).resolve()
        if index_file.parent.resolve() not in path.parents or path.stat().st_size != part['bytes'] or digest_file(path) != part['sha256']:
            raise ValueError('Prediction journal hash or path differs')
        lines = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(lines) != part['rows']:
            raise ValueError('Prediction journal row count differs')
        records.extend(lines)
    if len(records) != index['count']:
        raise ValueError('Prediction journal count differs')
    links = json.loads((trial / 'stream-prediction-sources.json').read_text())
    captures = {item['frame_index']: item for item in
                (json.loads(line) for line in (trial / 'continuous/capture.jsonl').read_text().splitlines() if line.strip())}
    scope = json.loads((trial / 'scope.json').read_text())
    if scope['checkpoint_sha256'] != PARENT_SHA:
        raise ValueError('Historical trial was not recorded with the parent checkpoint')
    selected = {}
    for number in (14, 15):
        chunk_file = trial / f'chunk-{number:05d}.json'
        chunk = json.loads(chunk_file.read_text())
        matches = [item for item in records if item['request']['id'] == chunk['request']['id']]
        if len(matches) != 1:
            raise ValueError('Historical request is missing or duplicated')
        record = matches[0]
        request, reply = record['request'], record['reply']
        if (not all(request.get(k) == v for k, v in chunk['request'].items())
                or reply != chunk['reply'] or reply['checkpoint_sha256'] != PARENT_SHA):
            raise ValueError('Historical chunk disagrees with its journal')
        jpeg = base64.b64decode(request['rgb_jpeg_base64'], validate=True)
        jpeg_hash = digest_bytes(jpeg)
        if jpeg_hash != request['rgb_jpeg']['sha256'] or len(jpeg) != request['rgb_jpeg']['bytes']:
            raise ValueError('Historical request JPEG hash differs')
        matching_links = [item for item in links if item['request_id'] == request['id']
                          and item['request_jpeg_hash_verified']]
        if len(matching_links) != 1:
            raise ValueError('Historical request has no unique capture link')
        link = matching_links[0]
        capture_path = (trial / 'continuous' / captures[link['frame_index']]['rgb_file']).resolve()
        if ((trial / 'continuous').resolve() not in capture_path.parents
                or link['rgb_jpeg_sha256'] != jpeg_hash
                or link['source_timestamp_monotonic'] != request['source_time']
                or capture_path.read_bytes() != jpeg):
            raise ValueError('Historical request JPEG differs from captured bytes')
        selected[number] = {'request': request, 'historical_reply': reply, 'captures': captures,
                            'evidence': {'chunk_sha256': digest_file(chunk_file),
                                         'request_id': request['id'], 'jpeg_sha256': jpeg_hash,
                                         'jpeg_bytes': len(jpeg), 'capture_frame_index': link['frame_index'],
                                         'capture_file': str(capture_path), 'jpeg_equals_capture_file': True}}
    return selected, {'prediction_index_sha256': digest_file(index_file),
                      'prediction_sources_sha256': digest_file(trial / 'stream-prediction-sources.json'),
                      'capture_journal_sha256': digest_file(trial / 'continuous/capture.jsonl'),
                      'scope_sha256': digest_file(trial / 'scope.json')}


def tensor_evidence(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': digest_bytes(raw)}


def batch_evidence(batch):
    import torch
    result = {}
    for key, value in sorted(batch.items()):
        if torch.is_tensor(value):
            result[key] = tensor_evidence(value)
        else:
            result[key] = {'json_sha256': json_hash(value)}
    return {'components': result, 'sha256': json_hash(result)}


class CaptureDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.sample = None

    def __getitem__(self, index):
        self.sample = self.dataset[index]
        return self.sample


def infer_batch(model, service, batch, state, seed):
    t = model.torch
    t.manual_seed(seed)
    model.reset()
    t.cuda.synchronize()
    start = time.monotonic()
    with t.inference_mode(), t.autocast('cuda', dtype=t.bfloat16):
        raw = model.policy.predict_action_chunk(batch)
        actions = model.post(raw.detach().clone()).cpu().float()
        actions = service.actions_for_wire(actions, state, model.relative)
    t.cuda.synchronize()
    if tuple(actions.shape) != (1, 50, 7) or not bool(t.isfinite(actions).all()):
        raise ValueError('Invalid predicted action chunk')
    return actions[0, :10].tolist(), time.monotonic() - start


def evaluate_hover(output, trainer, model, service, rows, episodes, dataset, proof, profile, draws, base_seed):
    evidence = json.loads(HOVER_EVIDENCE.read_text(encoding='utf-8'))
    cases = select_hover_cases(rows, episodes, proof, evidence)
    directions = {}
    for case in cases:
        if case['kind'] != 'hold_end':
            continue
        targets, valid = trainer.chunk_for_anchor(rows, episodes[case['episode']], case['position'])
        if not all(valid[:10]):
            raise ValueError('Hover return has fewer than ten real targets')
        state = rows[case['row_index']]['observation.state']
        delta = [float(targets[9][j]) - float(state[j]) for j in range(6)]
        magnitude = math.sqrt(sum(x*x for x in delta))
        if magnitude == 0:
            raise ValueError('Recorded hover return has no joint direction')
        directions[(case['source_index'], case['hold_index'])] = [x/magnitude for x in delta]
    capture = CaptureDataset(dataset)
    with (output / 'hover_cases.jsonl').open('x', encoding='utf-8') as progress:
        for case in cases:
            index, ep, pos = case['row_index'], case['episode'], case['position']
            state = [float(x) for x in rows[index]['observation.state']]
            targets, valid = trainer.chunk_for_anchor(rows, episodes[ep], pos)
            if not all(valid[:10]):
                raise ValueError('Hover anchor has fewer than ten real targets')
            batch, mask = known_case_batch(trainer, model, index, rows, episodes, capture, profile)
            if mask is not None or tuple(batch['observation.state'].shape) != (1, 32):
                raise ValueError('Unexpected hover prediction batch')
            sample = capture.sample
            if 'evidence_task' in case and sample['task'] != case.pop('evidence_task'):
                raise ValueError('Hover task differs from the recorded evidence')
            direction = directions[(case['source_index'], case['hold_index'])]
            case.update(state=state, task=sample['task'], gt_first10=targets[:10],
                        return_direction_unit=direction, draws=[])
            if profile == 'state-history-v1':
                case['observation_history'] = rows[index]['_observation_history']
            for seed in seeds_for_case(case['case_index'], draws, base_seed):
                actions, elapsed = infer_batch(model, service, batch, state, seed)
                case['draws'].append({'seed': seed, 'elapsed_seconds': elapsed, 'actions_first10': actions,
                                      'metrics': hover_joint_metrics(actions, targets[:10], state,
                                                                     direction)})
            progress.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + '\n')
            progress.flush()
            print(json.dumps({'status': 'hover_case', 'case_index': case['case_index'],
                              'row': index, 'source': case['source_index']}), flush=True)
    result = {'schema': 'fr5-hover-offline-diagnostic-v1', 'checkpoint_manifest_sha256': model.info['checkpoint_sha256'],
              'observation_context_profile': profile, 'evidence': str(HOVER_EVIDENCE),
              'cases_file': str(output / 'hover_cases.jsonl'), 'draws_per_case': draws,
              'seed_formula': 'base_seed + case_index * 1000 + draw_index', 'base_seed': base_seed,
              'return_direction_definition': 'unit vector from hold-end observation to its demonstrated tenth joint target',
              'projection_definition': 'tenth predicted or demonstrated joint target minus its own observation, projected onto return direction, degrees',
              'pairs': summarize_hover_pairs(cases),
              'limitations': ['Offline recorded-target diagnostic, not task success or an online threshold.',
                              'Hold start and end use distinct per-case seeds; their draw averages are descriptive.',
                              'Source 7 is the only held-out physical source.']}
    write_json_exclusive(output / 'hover_report.json', result)
    return result


def write_json_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def resolve_paths(checkpoint, dataset, output, audit_root, training_log):
    checkpoint, dataset, output, audit_root, training_log = (
        Path(path).resolve() for path in (checkpoint, dataset, output, audit_root, training_log))
    data_root = ROOT.resolve()
    audits = (ROOT / 'audits').resolve()
    if (data_root not in audits.parents or data_root not in checkpoint.parents or data_root not in dataset.parents
            or audits not in audit_root.parents or audits not in training_log.parents
            or audit_root not in output.parents):
        raise ValueError('Require data4 inputs, audit root and training log under data4/audits, and output under audit root')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    return checkpoint, dataset, output, audit_root, training_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--audit-root', type=Path, default=AUDIT_ROOT)
    parser.add_argument('--training-log', type=Path, default=TRAINING_LOG)
    parser.add_argument('--draws', type=int, default=8)
    parser.add_argument('--base-seed', type=int, default=BASE_SEED)
    args = parser.parse_args()
    checkpoint, dataset_path, output, audit_root, training_log = resolve_paths(
        args.checkpoint, args.dataset, args.output, args.audit_root, args.training_log)
    if (args.gpu < 0 or not 1 <= args.draws <= SEED_STRIDE or args.base_seed < 0
            or len(args.expected_sha) != 64 or any(c not in '0123456789abcdef' for c in args.expected_sha)):
        raise ValueError('Require a nonnegative GPU index and a valid SHA/draw count')
    manifest_sha = digest_file(checkpoint / 'stream-manifest.json')
    if manifest_sha != args.expected_sha:
        raise ValueError('Checkpoint manifest SHA mismatch')
    gradient = require_first_gradient(training_log)
    before = gpu_snapshot(args.gpu)
    require_free_gpu(before, expected_index=args.gpu)
    trial, trial_evidence = load_trial()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(checkpoint))
    import serve_stream_pi05 as service
    if Path(service.__file__).resolve() != (checkpoint / 'serve_stream_pi05.py').resolve():
        raise RuntimeError('Did not load sealed checkpoint service')
    require_free_gpu(gpu_snapshot(args.gpu), expected_index=args.gpu)
    print(json.dumps({'status': 'loading', 'pid': os.getpid(), 'checkpoint_sha256': manifest_sha,
                      'gpu_before': before}), flush=True)
    model = service.Inference(checkpoint, gpu=args.gpu)
    try:
        torch = model.torch
        torch.set_num_threads(4)
        if (model.info['checkpoint_sha256'] != manifest_sha
                or model.info['image_profile'] != 'workspace-only-v1'
                or model.info['action_representation'] != 'joint-delta-from-anchor-v1'
                or model.info['manifest']['dataset_fingerprint'] != DATASET_SHA
                or torch.cuda.device_count() != 1):
            raise ValueError('Checkpoint identity, dataset, or inference contract differs')
        trainer = model.training
        context_profile = model.info.get('observation_context_profile', 'none')
        if context_profile not in ('none', 'state-history-v1'):
            raise ValueError('Unknown checkpoint observation context profile')
        rows, episodes, dataset, proof, fingerprint = trainer.read_dataset(dataset_path)
        train_eps, val_eps, heldout = trainer.split_sources(rows, episodes)
        events = trainer.transition_events(rows, episodes)
        if (fingerprint != DATASET_SHA or len(proof['sources']) != 8 or len(train_eps) != 7
                or len(val_eps) != 1 or heldout != ['7'] or len(events['close']) != 10
                or len(events['open']) != 10):
            raise ValueError('Dataset fingerprint, source split, or transition counts differ')
        cases = select_known_cases(rows, episodes, events, trial[15]['request']['state'])
        if (len(cases) != 36 or not any(c['kind'] == 'approach_open' and c['source_index'] == 1
                                        and c['row_index'] == 702 for c in cases)):
            raise ValueError('Real anchor selection or source-1 approach does not match expected evidence')
        print(json.dumps({'status': 'dataset_verified', 'fingerprint': fingerprint,
                          'known_cases': len(cases), 'heldout_sources': heldout,
                          'approach_rows': [c['row_index'] for c in cases if c['kind'] == 'approach_open']}), flush=True)
        capture = CaptureDataset(dataset)
        finished = []
        with (output / 'cases.jsonl').open('x', encoding='utf-8') as progress:
            for case in cases:
                index, ep, position = case['row_index'], case['episode'], case['position']
                state = [float(v) for v in rows[index]['observation.state']]
                targets, valid = trainer.chunk_for_anchor(rows, episodes[ep], position)
                if not all(valid[:10]):
                    raise ValueError('First ten demonstrated targets are not complete')
                batch, mask = known_case_batch(trainer, model, index, rows, episodes, capture, context_profile)
                if mask is not None or tuple(batch['observation.state'].shape) != (1, 32) or batch.get('action') is not None:
                    raise ValueError('Unexpected model prediction batch')
                sample = capture.sample
                case.update(state=state, task=sample['task'], gt_first10=targets[:10],
                            gt_first10_gripper=[float(x[6]) for x in targets[:10]],
                            input={'origin': 'sealed trainer batch_for; decoded training video without JPEG re-encoding',
                                   'decoded_front_image': tensor_evidence(sample['observation.images.front']),
                                   'raw_state_and_task_sha256': json_hash({'state': state, 'task': sample['task']}),
                                   'model_batch': batch_evidence(batch)}, draws=[])
                if context_profile == 'state-history-v1':
                    case['observation_history'] = rows[index]['_observation_history']
                for seed in seeds_for_case(case['case_index'], args.draws, args.base_seed):
                    actions, elapsed = infer_batch(model, service, batch, state, seed)
                    case['draws'].append({'seed': seed, 'elapsed_seconds': elapsed,
                                          'actions_first10': actions,
                                          'gripper_first10': [x[6] for x in actions],
                                          'metrics': grip_metrics(actions, targets[:10])})
                if batch_evidence(batch) != case['input']['model_batch']:
                    raise RuntimeError('Model prediction mutated the recorded input batch')
                progress.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + '\n')
                progress.flush()
                finished.append(case)
                print(json.dumps({'status': 'case', 'completed': len(finished), 'total': 38,
                                  'kind': case['kind'], 'source': case['source_index'], 'row': index}), flush=True)
            negative = []
            for number in (14, 15):
                entry = trial[number]
                request = dict(entry['request'])
                history_provenance = None
                if context_profile == 'state-history-v1':
                    history, history_provenance = reconstruct_capture_history(
                        list(entry['captures'].values()), request, entry['evidence']['capture_frame_index'])
                    request['observation_history'] = history
                _, task, state, rgb = service.validate_request(request)
                if context_profile == 'state-history-v1':
                    pre_batch = service.prepare_request_batch(trainer, model.pre, state, rgb, task,
                        model.info['image_profile'], observation_context_profile=context_profile, history=history)
                else:
                    pre_batch = service.prepare_request_batch(trainer, model.pre, state, rgb, task, model.info['image_profile'])
                case_index = len(cases) + len(negative)
                case = {'kind': 'historical_offline_negative', 'case_index': case_index,
                        'chunk': number, 'split': 'not_dataset_no_joint_gt', 'evidence': entry['evidence'],
                        'state': state, 'task': task, 'joint_ground_truth_available': False,
                        'training_source': False,
                        'input': {'jpeg_sha256': entry['evidence']['jpeg_sha256'],
                                  'decoded_rgb_sha256': digest_bytes(rgb.tobytes()),
                                  'raw_state_and_task_sha256': json_hash({'state': state, 'task': task}),
                                  'model_batch_reconstructed_with_sealed_service': batch_evidence(pre_batch)},
                        'historical_parent_gripper_first10': [a[6] for a in entry['historical_reply']['actions'][:10]],
                        'draws': []}
                if context_profile == 'state-history-v1':
                    case['observation_history'] = history
                    case['history_capture_provenance'] = history_provenance
                for seed in seeds_for_case(case_index, args.draws, args.base_seed):
                    torch.manual_seed(seed)
                    model.reset()
                    answer = model.predict(request)
                    if answer['checkpoint_sha256'] != manifest_sha or answer['image_profile'] != model.info['image_profile']:
                        raise ValueError('Historical replay used a different checkpoint or profile')
                    if context_profile != 'none' and answer.get('observation_context_profile') != context_profile:
                        raise ValueError('Historical replay used a different observation context profile')
                    actions = answer['actions'][:10]
                    case['draws'].append({'seed': seed, 'elapsed_seconds': answer['inference_seconds'],
                                          'actions_first10': actions, 'gripper_first10': [x[6] for x in actions]})
                progress.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + '\n')
                progress.flush()
                negative.append(case)
                print(json.dumps({'status': 'case', 'completed': len(finished) + len(negative),
                                  'total': 38, 'kind': case['kind'], 'chunk': number}), flush=True)
        manifest = model.info['manifest']
        weights = {name: {'sha256': item['sha256'], 'bytes': item['bytes']} for name, item in manifest['files'].items()
                   if name.endswith('.safetensors')}
        weights['base_model'] = manifest['base']
        report = {'schema': 'fr5-conditioned-offline-comparison-v1',
                  'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'hostname': os.uname().nodename, 'pid': os.getpid(), 'python': sys.executable,
                  'checkpoint': str(checkpoint), 'checkpoint_manifest_sha256': manifest_sha,
                  'weights': weights, 'checkpoint_steps_completed': manifest['steps_completed'],
                  'sealed_service': str(checkpoint / 'serve_stream_pi05.py'),
                  'sealed_trainer': str(checkpoint / 'train_stream_pi05.py'),
                  'dataset': str(dataset_path), 'dataset_fingerprint': fingerprint,
                  'gpu_before': before, 'formal_first_gradient': gradient, 'training_log': str(training_log),
                  'source_count': 8, 'training_sources': 7, 'validation_sources': 1,
                  'heldout_sources': heldout, 'event_counts': {k: len(v) for k, v in events.items()},
                  'base_seed': args.base_seed, 'seed_formula': 'base_seed + case_index * 1000 + draw_index',
                  'draws_per_case': args.draws, 'image_profile': model.info['image_profile'],
                  'observation_context_profile': context_profile,
                  'action_representation': model.info['action_representation'],
                  'match_definition': 'each predicted gripper value is strictly closer to its recorded target (0.76 or 1.0) than to the other; no clipping',
                  'trial_evidence': trial_evidence, 'cases': finished, 'negative_cases': negative,
                  'aggregates': aggregate(finished), 'robot_or_camera_access': False, 'training_performed': False,
                  'limitations': ['Open-loop target agreement is not a grasp or whole-task success rate.',
                                  'Draws from the same anchor and anchors from one source are correlated.',
                                  'The held-out split has one physical source, not an independent population.',
                                  'Historical trial replays have no joint ground truth and are not training sources.',
                                  'Historical replay seeds are new and do not reproduce the original random sequence.',
                                  'Joint errors are diagnostics, not primary task success measures.']}
        write_json_exclusive(output / 'report.json', report)
        evaluate_hover(output, trainer, model, service, rows, episodes, dataset, proof,
                       context_profile, args.draws, args.base_seed)
        print(json.dumps({'status': 'complete', 'report': str(output / 'report.json'),
                          'known_cases': len(finished), 'negative_cases': len(negative)}), flush=True)
    finally:
        model.reset()
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({'status': 'releasing_gpu_process_will_exit', 'pid': os.getpid()}), flush=True)


if __name__ == '__main__':
    main()
