#!/usr/bin/env python3
"""Prepare shared terminal evaluation phases from recorded CPU evidence."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys
from types import ModuleType

sys.dont_write_bytecode = True

ROOT = Path('/home/data4/hts/demo1/fr5-learning')
PHASE = ('state', 'phase')
CUP_FIELDS = ('camera_serial', 'base_xyz_mm', 'rim_xyz_mm', 'table_normal', 'table_offset_mm')
CLOCK_FIELDS = ('contact_time', 'hover_start', 'hover_seconds', 'setdown_since', 'setdown_last',
                'was_released', 'retreat_peak')
PRESERVED = ('original_jpeg', 'camera_timestamp', 'robot_and_gripper_feedback', 'history',
             'task', 'targets', 'tracking', 'contact_clock', 'hover_clock', 'support_gap',
             'setdown_clock', 'other_timing_fields', 'failed_trial_role')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def differences(original, corrected, path=()):
    """Compare JSON values exactly, including missing fields and value types."""
    if isinstance(original, dict) and isinstance(corrected, dict):
        for key in sorted(original.keys() | corrected.keys()):
            if key not in original or key not in corrected:
                yield path + (key,)
            else:
                yield from differences(original[key], corrected[key], path + (key,))
    elif isinstance(original, list) and isinstance(corrected, list):
        if len(original) != len(corrected):
            yield path
        else:
            for index, (a, b) in enumerate(zip(original, corrected)):
                yield from differences(a, b, path + (str(index),))
    elif type(original) is not type(corrected) or original != corrected:
        yield path


def observer(case):
    if 'observers' in case:
        saved = case['observers']['setdown']
        return saved['tracking'], saved['timing'], saved['model_timing']
    return case['tracking'], case['timing_setdown'], case['timing_setdown']['state']


def phase_paths(case):
    if 'observers' in case:
        return (('observers', 'setdown', 'timing', *PHASE),
                ('observers', 'setdown', 'model_timing', 'phase'))
    return (('timing_setdown', *PHASE),)


def set_phase(case, phase):
    for path in phase_paths(case):
        value = case
        for key in path[:-1]:
            value = value[key]
        value[path[-1]] = phase


def verify_case(original, corrected):
    paths = list(differences(original, corrected))
    outside_phase = [path for path in paths if path not in phase_paths(original)]
    if outside_phase:
        raise ValueError(f"{original['case_id']}: changed /" + '/'.join(outside_phase[0]))
    if paths and (observer(original)[2]['phase'], observer(corrected)[2]['phase']) != ('at_pickup', 'released'):
        raise ValueError(f"{original['case_id']}: only recorded at_pickup to released changes are allowed")
    return paths


def load_audit(directory):
    audit = read_json(directory / 'summary.json')
    if (audit.get('verified') is not True or audit['baseline_difference_counts']
            or audit['release_scope_violations'] or any(audit['internal_difference_counts'].values())
            or any(count for path, count in audit['output_difference_counts'].items() if path != 'state.phase')
            or not audit['method']['aligned'] or not audit['method']['setdown']):
        raise ValueError('Expected a verified phase-only aligned setdown audit')
    original = read_json(audit['method']['original_summary'])
    if original['complete'] is not True or original['failures']:
        raise ValueError('Original-frame tracking evidence is incomplete')
    sources = {str(Path(row['trial'])): row for row in original['results']}
    if len(sources) != len(original['results']):
        raise ValueError('Original-frame evidence repeats a source root')
    changes = {}
    for row in read_jsonl(directory / 'changed-frames.jsonl'):
        key = row['source'], row['frame']
        if (key in changes or row['differences'] != {'state.phase': ['at_pickup', 'released']}
                or row['gripper_feedback_valid'] is not True or row['native_contact'] != 0
                or row['was_released'] is not True):
            raise ValueError('Invalid or repeated released-phase evidence')
        changes[key] = row
    if len(changes) != audit['changed_phase_frames']:
        raise ValueError('Released-phase evidence does not cover the verified audit')
    return audit, sources, changes


def audited_source(source, selected, changes):
    records = {}
    for line_number, row in enumerate(read_jsonl(source['evidence']), 1):
        capture = row['capture']
        frame = capture['frame_index']
        if frame not in selected:
            continue
        before = row['timing']['new']
        after = deepcopy(before)
        change = changes.get((source['id'], frame))
        if change is not None:
            if (change['camera_monotonic'] != capture['camera']['source_timestamp_monotonic']
                    or change['feedback_monotonic'] != capture['robot']['received_monotonic']
                    or before['state']['phase'] != 'at_pickup'
                    or capture['gripper_feedback_valid'] is not True
                    or row['tracking']['new']['native_contact'] != 0):
                raise ValueError('Released-phase evidence and original capture differ')
            after['state']['phase'] = 'released'
        if frame in records:
            raise ValueError('Original-frame evidence repeats a frame')
        records[frame] = {'capture': capture, 'tracking': row['tracking']['new'],
                          'before': before, 'after': after,
                          'was_released': after['state']['phase'] == 'released',
                          'trace_line_1based': line_number}
    return dict(source, records=records, start_frame=0, method='verified_original_frame_audit',
                rgbd_recomputed=False, frames_replayed=0)


def timer_class(path):
    module = ModuleType(path.stem)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module.CupTiming


def causal_source(source, selected, rows, before_class, after_class, *, start=0, rgbd=False):
    before_timer = before_class(source['initial_cup'], aligned=True, setdown=True)
    after_timer = after_class(source['initial_cup'], aligned=True, setdown=True)
    records, count, changes = {}, 0, []
    for line_number, row in enumerate(rows, 1):
        capture = row['capture']
        frame = capture['frame_index']
        if frame < start:
            continue
        if frame > max(selected):
            break
        tracking = row['tracking']['new']
        before = before_timer.update(capture['camera'], capture['robot'], tracking)
        after = after_timer.update(capture['camera'], capture['robot'], tracking)
        if 'timing' in row and list(differences(row['timing']['new'], before)):
            raise ValueError(f'Original timing replay differs at frame {frame}')
        paths = list(differences(before, after))
        if any(path != PHASE for path in paths) or any(
                getattr(before_timer, key) != getattr(after_timer, key) for key in CLOCK_FIELDS):
            raise ValueError(f'Contact, hover, support or setdown clock changed at frame {frame}')
        if paths:
            if ((before['state']['phase'], after['state']['phase']) != ('at_pickup', 'released')
                    or capture['gripper_feedback_valid'] is not True or tracking['native_contact'] != 0
                    or not after_timer.was_released):
                raise ValueError(f'Phase changed outside a recorded release at frame {frame}')
            changes.append(frame)
        count += 1
        if frame in selected:
            if frame in records:
                raise ValueError('Causal evidence repeats a selected frame')
            records[frame] = {'capture': capture, 'tracking': tracking, 'before': before, 'after': after,
                              'was_released': bool(after_timer.was_released),
                              'trace_line_1based': line_number}
    return dict(source, records=records, start_frame=start,
                method='causal_rgbd_replay' if rgbd else 'causal_saved_tracking_replay',
                rgbd_recomputed=rgbd, frames_replayed=count, changed_phase_frames=changes,
                nonphase_timing_changes=0, internal_clock_changes=0)


def capture_tracking(root, captures, cup, calibration, start, last):
    """Replay the recorded tracker through the last requested frame on CPU."""
    import cv2
    import numpy as np
    from fr5_jev.cup_tracking import CupTracker

    cv2.setNumThreads(1)
    tracker = CupTracker(cup, calibration['base_from_color_camera_mm'])
    for capture in captures:
        frame = capture['frame_index']
        if frame < start:
            continue
        if frame > last:
            break
        image = cv2.imread(str(root / 'continuous' / capture['rgb_file']), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Missing original RGB frame {frame}')
        with np.load(root / 'continuous' / capture['depth_file'], allow_pickle=False) as depth:
            tracking = tracker.update(image, depth['depth_mm'], capture['camera'], capture['robot'],
                                      gripper_valid=bool(capture['gripper_feedback_valid']))
        yield {'capture': capture, 'tracking': {'new': tracking}}


def collect_evidence(inputs, audit_dir, source_root):
    audit, saved, changes = load_audit(audit_dir)
    groups = defaultdict(list)
    for report in inputs.values():
        for case in report['cases']:
            groups[str(Path(case['source_root']))].append(case)
    evidence, source_errors = {}, {}
    classes = None
    for root_name, cases in groups.items():
        selected = {case['source_frame_index'] for case in cases}
        source = saved.get(root_name)
        try:
            if source is not None and source['id'].startswith('source-'):
                evidence[root_name] = audited_source(source, selected, changes)
            else:
                if classes is None:
                    classes = (timer_class(audit_dir / 'cup_timing.before.py'),
                               timer_class(audit_dir / 'cup_timing.after.py'))
                if source is not None:
                    rows = read_jsonl(source['evidence'])
                    evidence[root_name] = causal_source(source, selected, rows, *classes)
                else:
                    starts = {case.get('tracker_start_frame_index', 0) for case in cases}
                    if len(starts) != 1:
                        raise ValueError('Cases have different original tracker starts')
                    root = Path(root_name)
                    cup = cases[0]['initial_cup']
                    calibration_root = next(Path(case['source_root']) for report in inputs.values()
                                            for case in report['cases'] if case.get('source_index') == 7)
                    calibration = read_json(calibration_root / 'source-observation.json')['calibration']
                    if calibration['camera_serial'] != cup['camera_serial']:
                        raise ValueError('Recorded camera and calibration differ')
                    captures = read_jsonl(root / 'continuous/capture.jsonl')
                    start = starts.pop()
                    rows = capture_tracking(root, captures, cup, calibration, start, max(selected))
                    source = {'id': root.parent.name, 'trial': root_name, 'initial_cup': cup,
                              'evidence': str(root / 'continuous/capture.jsonl')}
                    evidence[root_name] = causal_source(source, selected, rows, *classes, start=start, rgbd=True)
            print(f"{root_name}: {len(evidence[root_name]['records'])} original observations matched", flush=True)
        except (FileNotFoundError, ValueError) as error:
            source_errors[root_name] = str(error)
    return audit, evidence, source_errors


def prepare_case(case, source, error=None):
    corrected = deepcopy(case)
    tracking, timing, model_timing = observer(case)
    old_phase = model_timing['phase']
    detail = {key: case.get(key) for key in ('case_id', 'group', 'kind', 'source_root', 'source_frame_index')}
    detail.update(source_time=case['request']['source_time'], old_phase=old_phase,
                  new_phase=old_phase, status='unassessed', phase_changed=False, changed_paths=[])
    reasons = []
    if source is None:
        reasons.append(error or 'missing_source_evidence')
    else:
        record = source['records'].get(case['source_frame_index'])
        if record is None:
            reasons.append('missing_original_frame_evidence')
        else:
            capture, expected = record['capture'], record['after']
            if (case['request']['source_clock'] != 'host_monotonic'
                    or case['request']['source_time'] != capture['camera']['source_timestamp_monotonic']):
                reasons.append('source_frame_time_mismatch')
            if str(Path(case['source_root'])) != str(Path(source['trial'])):
                reasons.append('source_root_mismatch')
            if case.get('tracker_start_frame_index', 0) != source['start_frame']:
                reasons.append('tracker_start_mismatch')
            if case.get('source_index') is not None and source['id'] != 'source-' + str(case['source_index']):
                reasons.append('source_index_mismatch')
            if any(case['initial_cup'][key] != source['initial_cup'][key] for key in CUP_FIELDS):
                reasons.append('initial_cup_mismatch')
            if list(differences(tracking, record['tracking'])):
                reasons.append('tracking_mismatch')
            if list(differences(timing['state'], model_timing)):
                reasons.append('model_timing_mismatch')
            if any(path != PHASE for path in differences(timing, expected)):
                reasons.append('nonphase_timing_mismatch')
            if old_phase not in (record['before']['state']['phase'], expected['state']['phase']):
                reasons.append('original_phase_mismatch')
            if old_phase != expected['state']['phase'] and (
                    (old_phase, expected['state']['phase']) != ('at_pickup', 'released')
                    or capture['gripper_feedback_valid'] is not True
                    or record['tracking']['native_contact'] != 0 or record['was_released'] is not True):
                reasons.append('missing_confirmed_release')
            detail.update(evidence=source['evidence'], evidence_method=source['method'],
                          trace_line_1based=record['trace_line_1based'],
                          camera_monotonic=capture['camera']['source_timestamp_monotonic'],
                          feedback_monotonic=capture['robot']['received_monotonic'],
                          gripper_feedback_valid=capture['gripper_feedback_valid'],
                          gripper_position_percent=capture['robot']['gripper_position'],
                          native_contact=record['tracking']['native_contact'], was_released=record['was_released'])
            if not reasons:
                set_phase(corrected, expected['state']['phase'])
                paths = verify_case(case, corrected)
                detail.update(status='assessed', new_phase=expected['state']['phase'],
                              phase_changed=bool(paths), changed_paths=['/' + '/'.join(path) for path in paths])
    detail['unassessed_reasons'] = reasons
    verify_case(case, corrected)
    return corrected, detail


def prepare_inputs(inputs, evidence, source_errors, audit_dir, output_dir, input_paths):
    outputs, details, counts = {}, [], {}
    for name, original in inputs.items():
        report = deepcopy(original)
        rows = []
        for index, case in enumerate(original['cases']):
            root = str(Path(case['source_root']))
            corrected, detail = prepare_case(case, evidence.get(root), source_errors.get(root))
            report['cases'][index] = corrected
            detail.update(input_name=name, case_index=index)
            rows.append(detail)
        unassessed = [row['case_id'] for row in rows if row['status'] == 'unassessed']
        counter = {'case_count': len(rows), 'assessed_cases': len(rows) - len(unassessed),
                   'unassessed_cases': len(unassessed), 'phase_changed_cases': sum(row['phase_changed'] for row in rows),
                   'phase_changed_values': sum(len(row['changed_paths']) for row in rows),
                   'released_cases': sum(row['status'] == 'assessed' and row['new_phase'] == 'released' for row in rows),
                   'phase_counts': dict(Counter(row['new_phase'] for row in rows)),
                   'unassessed_case_ids': unassessed, 'input': str(input_paths[name]),
                   'output': str(output_dir / name), 'preserved': {field: True for field in PRESERVED}}
        report['terminal_phase_evaluation'] = {
            'phase_audit': str(audit_dir / 'summary.json'), 'original_input': str(input_paths[name]),
            'evidence': str(output_dir / 'phase-evidence.jsonl'), 'ready': not unassessed,
            'assessed_cases': counter['assessed_cases'], 'unassessed_case_ids': unassessed,
            'shared_for': ['clean', 'terminal'],
        }
        outputs[name], counts[name] = report, counter
        details.extend(rows)
    return outputs, details, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Frozen tracker source used for missing traces')
    parser.add_argument('--phase-audit', type=Path, default=ROOT / 'audits/cup-timing-released-stage-20261116-v1')
    original = ROOT / 'audits/pi05-cup-precontact-close-20261115-v4'
    for name in ('heldout', 'fit', 'trial01', 'trial02'):
        parser.add_argument('--' + name + '-inputs', type=Path, default=original / (name + '-inputs.json'))
    parser.add_argument('--training-support-inputs', type=Path,
                        default=ROOT / 'audits/pi05-cup-approach-support-20261115-v4/inputs.json')
    parser.add_argument('--recovered-feedback-inputs', type=Path,
                        default=ROOT / 'audits/pi05-cup-fixed-clock-recovered-pgi-inputs-corrected-phase-20261116-v1.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    input_paths = {name + '-inputs.json': getattr(args, name + '_inputs')
                   for name in ('heldout', 'fit', 'trial01', 'trial02')}
    input_paths.update({'training-support-inputs.json': args.training_support_inputs,
                        'recovered-feedback-inputs.json': args.recovered_feedback_inputs})
    targets = [args.output_dir / name for name in (*input_paths, 'phase-evidence.jsonl', 'summary.json')]
    if any(path.exists() for path in targets):
        raise FileExistsError('Terminal evaluation outputs already exist')
    inputs = {name: read_json(path) for name, path in input_paths.items()}
    audit, evidence, errors = collect_evidence(inputs, args.phase_audit, args.source_root)
    outputs, details, counts = prepare_inputs(inputs, evidence, errors, args.phase_audit, args.output_dir, input_paths)
    total = {key: sum(row[key] for row in counts.values()) for key in
             ('case_count', 'assessed_cases', 'unassessed_cases', 'phase_changed_cases', 'phase_changed_values', 'released_cases')}
    summary = {
        'schema': 'fr5-cup-terminal-evaluation-inputs-v9', 'status': 'prepared' if not total['unassessed_cases'] else 'partial',
        **total, 'outputs': counts, 'shared_for': ['clean', 'terminal'],
        'phase_evidence': str(args.output_dir / 'phase-evidence.jsonl'),
        'original_phase_audit': {'path': str(args.phase_audit / 'summary.json'), 'frames': audit['frames'],
                                'sources_count': audit['sources_count'], 'changed_phase_frames': audit['changed_phase_frames'],
                                'output_difference_counts': audit['output_difference_counts'],
                                'internal_difference_counts': audit['internal_difference_counts'], 'reused': True},
        'source_evidence': {root: {key: value for key, value in source.items()
                                  if key in ('id', 'evidence', 'method', 'rgbd_recomputed', 'frames_replayed',
                                             'changed_phase_frames', 'nonphase_timing_changes', 'internal_clock_changes')}
                            for root, source in evidence.items()},
        'unassessed_source_errors': errors, 'preserved': {field: True for field in PRESERVED},
        'failed_trial_cases': sum(case.get('group') == 'failed_trial_evaluation_only'
                                  for report in inputs.values() for case in report['cases']),
        'failed_trial_use': 'evaluation_only', 'cpu_only': True, 'policy_inference': False,
        'robot_camera_connections': 0, 'robot_commands': 0,
        'commands': {'prepare': shlex.join([sys.executable, '-B', *sys.argv])},
        'limitations': ['released 表示原帧的有效开爪阶段，不证明杯子放稳或抓杯成功。',
                        '策略正式评分与实机成功尚未验证。'],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in outputs.items():
        write_new(args.output_dir / name, report)
    with (args.output_dir / 'phase-evidence.jsonl').open('x', encoding='utf-8') as stream:
        for detail in details:
            stream.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + '\n')
    write_new(args.output_dir / 'summary.json', summary)
    print(json.dumps({'output_dir': str(args.output_dir), **total, 'status': summary['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
