"""Offline pilot clips from completed target commands; never relabel a failed episode.

Long supervisor pauses and uncertain commands are excluded. Every output action
comes from an actually submitted target, checked against its successful RPC log.
Clips from one trial are not independent trials. The source split reservation
is retained so an evaluation trial cannot silently enter training.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .robot_fairino import STATE_NAMES
from .feedback import gripper_feedback_valid
from .human_review import verify_success_review
from lerobot_camera_gemini.timing import camera_clock, command_clock, completion_clock, MONOTONIC

FPS = 6
# A grid cell is narrower than half a period on each side. Adjacent cells
# therefore cannot claim the same source timestamp. Exclude the ambiguous
# halfway boundary by one microsecond, matching the recorded audit precision.
GRID_BOUNDARY_EPSILON_S = 1e-6
MAX_GRID_ERROR_S = 1 / (2 * FPS) - GRID_BOUNDARY_EPSILON_S
MAX_SOURCE_GAP_S = .30
PHASES = ["close", "lift", "return", "release"]
SHUTTLE_PHASES = ["close", "lift", "shuttle_out", "shuttle_back", "return", "release"]
VARIED_PHASES = ['approach_clear','approach_near','approach_contact','close','lift_probe',
                 'lift_full','return','release','retreat_probe','retreat_clear','retreat_high']
TASKS = {"close": "Close the gripper around the prealigned cup",
         "lift": "Lift the grasped cup vertically",
         "return": "Lower the cup to its original table support",
         "release": "Release the supported cup",
         "shuttle_out": "Move the held cup10mm along base positive X at10mm lift",
         "shuttle_back": "Return the held cup horizontally to its original XY at10mm lift"}
TASKS.update(approach_clear='Move the open gripper to the clear approach height beside the cup',
             approach_near='Approach the exterior of the cup with the gripper fully open',
             approach_contact='Align the open fingers around the outside of the cup',
             lift_probe='Lift the grasped cup10mm and verify that it follows',
             lift_full='Lift the grasped cup50mm above its original table support',
             retreat_probe='Withdraw the open gripper10mm from the released supported cup',
             retreat_clear='Withdraw the open gripper clear of the cup',
             retreat_high='Raise the empty open gripper to its displaced starting pose')


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def phase_order(metadata):
    phases = metadata.get('phase_order', PHASES)
    if phases not in (PHASES, SHUTTLE_PHASES, VARIED_PHASES):
        raise ValueError('Unknown or reordered task phase schema')
    return list(phases)


def _validate_interrupted_prefix(episode, summary, files):
    """Accept only a fully recorded approach prefix stopped before dispatch."""
    root=Path(episode)/'sources/00'
    def checked(name):
        path=root/name
        expected=files.get(f'sources/00/{name}')
        if not path.is_file() or not expected or file_sha256(path)!=expected:
            raise ValueError('Missing or changed interrupted-prefix evidence')
        return path
    if summary.get('outcome')!='interrupted' or summary.get('released') is not False or not summary.get('failed'):
        raise ValueError('Source is not a guarded interrupted approach')
    continuation=json.loads(checked('continuation-state.json').read_text())
    outcomes=json.loads(checked('action-outcomes.json').read_text())
    journal=json.loads(checked('target-command-journal.json').read_text())
    if (continuation.get('outcome')!='interrupted_before_final_entry_not_success'
            or continuation.get('grasp_completed') is not False
            or 'before dispatch' not in str(continuation.get('issue','')).lower()):
        raise ValueError('Interrupted prefix lacks evidence of a pre-dispatch stop')
    command_files=sorted(root.glob('command-*.json'))
    for file in command_files:
        checked(file.name)
        record=json.loads(file.read_text())
        if (record.get('status')!='completed' or record.get('outcome',{}).get('return_code')!=0
                or record.get('command',{}).get('executed') is not True):
            raise ValueError('Interrupted prefix contains an uncertain or failed command')
    commands=validated_commands(root)
    validate_journal(journal,commands,MONOTONIC)
    if (len(journal)!=len(commands)+1 or set(commands)!={item['sequence'] for item in journal[1:]}
            or summary.get('commands')!=len(commands) or continuation.get('completed_arm_commands')!=len(commands)
            or not isinstance(outcomes,list) or len(outcomes)!=len(commands)
            or any(not item.get('id') or item.get('status')!='completed' for item in outcomes)
            or len({item['id'] for item in outcomes})!=len(outcomes)):
        raise ValueError('Interrupted prefix command, outcome or journal mismatch')
    ordered=[commands[item['sequence']] for item in journal[1:]]
    previous_end=None
    for row,outcome in zip(ordered,outcomes):
        if (row['method']!='MoveL' or row['phase'] not in VARIED_PHASES[:3]
                or not row.get('samples') or row.get('final_state',{}).get('motion_done')!=1
                or (row.get('planned_action',{}).get('id') is not None
                    and row['planned_action']['id']!=outcome['id'])):
            raise ValueError('Interrupted prefix lacks completed approach feedback')
        start=command_clock(row['command'],MONOTONIC);end=completion_clock(row,MONOTONIC)
        if previous_end is not None and start<previous_end:
            raise ValueError('Interrupted prefix command windows overlap')
        previous_end=end
        for sample in row['samples']:
            if any(sample.get(key)!=0 for key in ('main_code','sub_code','EmergencyStop','collisionState',
                    'safety_stop0_state','safety_stop1_state','gripper_fault','gripper_fault_id')):
                raise ValueError('Interrupted prefix contains fault feedback')
    stop=root/'STOP'
    if stop.exists() and checked('STOP').read_text().strip()!=summary['failed']:
        raise ValueError('Interrupted prefix stop reason changed')
    for name in ('verified-progress.json',):
        if (root/name).exists():checked(name)


def physical_components(episode, *, allow_pending_summary=False):
    """Identify source recordings, including the parts of a logical composite."""
    episode = Path(episode)
    scope_raw = (episode/'scope.json').read_bytes()
    scope = json.loads(scope_raw)
    summary_file=episode/'summary.json'
    if allow_pending_summary and not summary_file.is_file():
        summary_file=episode/'summary.pending-curation.json'
    summary_raw = summary_file.read_bytes()
    summary = json.loads(summary_raw)
    if scope.get('schema') != 'fr5-composite-logical-scope-v1':
        return [{'scope_sha256':hashlib.sha256(scope_raw).hexdigest(),
                 'summary_sha256':hashlib.sha256(summary_raw).hexdigest(),
                 'outcome':summary.get('outcome'),'split_role':scope.get('split_role')}]
    if (summary.get('schema')!='fr5-composite-logical-summary-v1'
            or summary.get('scope_sha256')!=hashlib.sha256(scope_raw).hexdigest()
            or summary.get('outcome')!='pending_review' or summary.get('failed')
            or summary.get('released') is not True or scope.get('phase_order')!=VARIED_PHASES
            or summary.get('completed_stages')!=VARIED_PHASES
            or summary.get('continuous_capture',{}).get('capture_is_continuous_across_segments') is not False):
        raise ValueError('Invalid logical composite summary or scope')
    manifest=json.loads((episode/'composition-manifest.json').read_text())
    hashes=scope.get('segment_scope_sha256')
    if (manifest.get('schema')!='fr5-composite-logical-episode-v1'
            or manifest.get('logical_episode_only') is not True
            or manifest.get('continuous_physical_recording') is not False
            or scope.get('capture_is_continuous_across_segments') is not False
            or not isinstance(hashes,list) or len(hashes)!=2
            or len(manifest.get('source_segments',[]))!=len(hashes)):
        raise ValueError('Incomplete logical composite provenance')
    file_list=manifest.get('source_files',[])
    files={item['output']:item['sha256'] for item in file_list}
    if len(files)!=len(file_list):raise ValueError('Duplicate composite source file identity')
    base=episode.resolve()
    for relative,digest in files.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid composite source file path')
        path=(episode/path).resolve()
        if base not in path.parents or not path.is_file() or file_sha256(path)!=digest:
            raise ValueError('Composite source file identity changed')
    frame_manifest=Path(manifest.get('frame_manifest',''))
    frame_path=(episode/frame_manifest).resolve()
    if (frame_manifest.is_absolute() or '..' in frame_manifest.parts or not frame_manifest.parts
            or base not in frame_path.parents or file_sha256(frame_path)!=manifest.get('frame_manifest_sha256')):
        raise ValueError('Composite frame manifest identity changed')
    components=[]
    for i, expected_scope_hash in enumerate(hashes):
        prefix=f'sources/{i:02d}'
        source_scope_raw=(episode/prefix/'scope.json').read_bytes()
        source_summary_raw=(episode/prefix/'summary.json').read_bytes()
        source_scope=json.loads(source_scope_raw);source_summary=json.loads(source_summary_raw)
        scope_hash=hashlib.sha256(source_scope_raw).hexdigest()
        summary_hash=hashlib.sha256(source_summary_raw).hexdigest()
        if (scope_hash!=expected_scope_hash or files.get(f'{prefix}/scope.json')!=scope_hash
                or files.get(f'{prefix}/summary.json')!=summary_hash
                or source_summary.get('scope_sha256')!=scope_hash
                or source_scope.get('split_role')!=scope.get('split_role')
                or source_summary.get('split_role')!=scope.get('split_role')
                or source_summary.get('outcome') not in ('interrupted','pending_review')):
            raise ValueError('Composite component identity, split or failure state is invalid')
        segment=manifest['source_segments'][i]
        if (segment.get('segment')!=i or segment.get('outcome')!=source_summary.get('outcome')
                or segment.get('failure')!=source_summary.get('failed')):
            raise ValueError('Composite segment outcome differs from source summary')
        if i==0:
            _validate_interrupted_prefix(episode,source_summary,files)
        elif (source_summary.get('failed') or source_summary.get('outcome')!='pending_review'
              or source_summary.get('released') is not True or source_summary.get('completed_stages')!=VARIED_PHASES
              or source_summary.get('continuous_capture',{}).get('capture_complete') is not True):
            raise ValueError('Composite contains a failed pickup/place source')
        components.append({'scope_sha256':scope_hash,'summary_sha256':summary_hash,
                           'outcome':source_summary.get('outcome'),'failure':source_summary.get('failed'),
                           'split_role':source_scope['split_role']})
    if len({part['summary_sha256'] for part in components})!=len(components):
        raise ValueError('Composite repeats the same source recording')
    return components


def composite_frame_provenance(episode, rows, image_rows):
    """Check remapped composite records against retained original capture logs."""
    episode=Path(episode)
    scope=json.loads((episode/'scope.json').read_text())
    if scope.get('schema')!='fr5-composite-logical-scope-v1':return None
    manifest=json.loads((episode/'composition-manifest.json').read_text())
    sequences={}
    for item in [*manifest['command_sequence_map'],*manifest['hold_sequence_map']]:
        key=(item['segment'],item['source_sequence'])
        if key in sequences:raise ValueError('Duplicate original composite command sequence')
        sequences[key]=item['output_sequence']
    originals={};original_images={}
    for i in range(len(manifest['source_segments'])):
        root=episode/f'sources/{i:02d}'
        raw=[json.loads(s) for s in (root/'lerobot/meta/capture.jsonl').read_text().splitlines()]
        images=[json.loads(s) for s in (root/'continuous/capture.jsonl').read_text().splitlines()]
        originals[i]={r['frame_index']:r for r in raw}
        original_images[i]={r['frame_index']:r for r in images}
        if len(originals[i])!=len(raw) or len(original_images[i])!=len(images):
            raise ValueError('Duplicate original composite frame index')
    frame_records={}
    with (episode/manifest['frame_manifest']).open() as file:
        for line in file:
            record=json.loads(line);key=(record['stream'],record['output_frame_index'])
            if key in frame_records:raise ValueError('Duplicate composite frame manifest index')
            frame_records[key]=record
    for row in rows:
        source=row['composition_source'];segment=source['segment'];index=source['frame_index']
        old=originals[segment][index];record=frame_records['action_capture',row['frame_index']]
        old_command=dict(old['command']);new_command=dict(row['command'])
        old_sequence=old_command.pop('sequence');new_sequence=new_command.pop('sequence')
        if (record['segment']!=segment or record['source_frame_index']!=index
                or old['camera']!=row['camera'] or old['robot']!=row['robot']
                or old.get('sync_clock')!=row.get('sync_clock') or old.get('sync_skew_s')!=row.get('sync_skew_s')
                or sequences.get((segment,old_sequence))!=new_sequence or old_command!=new_command):
            raise ValueError('Composite action observation differs from original source')
    image_hashes={}
    for row in image_rows:
        source=row['composition_source'];segment=source['segment'];index=source['frame_index']
        old=original_images[segment][index];record=frame_records['continuous',row['frame_index']]
        rgb=record['files']['rgb_file']
        if (record['segment']!=segment or record['source_frame_index']!=index
                or old['camera']!=row['camera'] or old['robot']!=row['robot']
                or old.get('sync_clock')!=row.get('sync_clock') or old.get('sync_skew_s')!=row.get('sync_skew_s')
                or rgb['source']!=old['rgb_file'] or rgb['output']!=str(Path('continuous')/row['rgb_file'])):
            raise ValueError('Composite image observation differs from original source')
        image_hashes[row['frame_index']]=rgb['sha256']
    for item in manifest['command_sequence_map']:
        segment=item['segment'];name=Path(item['source_file'])
        if len(name.parts)!=1 or not name.name.startswith('command-') or name.suffix!='.json':
            raise ValueError('Invalid composite command source path')
        old=json.loads((episode/f'sources/{segment:02d}'/name).read_text())
        current=json.loads((episode/f"command-{item['output_sequence']:04d}.json").read_text())
        if old['command']['sequence']!=item['source_sequence']:
            raise ValueError('Composite command source identity changed')
        old['command']['sequence']=item['output_sequence']
        old['composition_source']={'segment':segment,'file':item['source_file'],'sequence':item['source_sequence']}
        if old!=current:raise ValueError('Composite command differs from original command log')
    return image_hashes


def validated_commands(episode):
    result = {}
    for file in sorted(episode.glob("command-*.json")):
        row = json.loads(file.read_text())
        if row.get("status") != "completed" or row.get("outcome", {}).get("return_code") != 0:
            continue
        command = row["command"]
        target = np.asarray(command["target"], float)
        if target.shape != (7,) or not np.isfinite(target).all() or command.get("executed") is not True:
            raise ValueError("Invalid submitted target")
        if not 0 <= target[6] <= 100:
            raise ValueError("Gripper command outside 0..100 percent")
        if row["method"] == "MoveL":
            if not np.array_equal(target[:6], row["args"][0]):
                raise ValueError("Action target differs from the actual MoveL argument")
        elif row["method"] == "MoveGripper":
            if target[6] != row["args"][1]:
                raise ValueError("Gripper label differs from the actual command")
        else:
            raise ValueError("Unsupported action semantics")
        if command["sequence"] in result:
            raise ValueError("Duplicate command sequence")
        result[command["sequence"]] = {**row, "source_file": str(file)}
    if not result:
        raise ValueError("No completed target commands")
    return result


def validate_journal(journal, commands, basis):
    if not journal or journal[0].get('kind') != 'hold' or journal[0].get('executed') is not False:
        raise ValueError('Missing declared initial hold')
    ids = [c['sequence'] for c in journal]
    times = [command_clock(c, basis) for c in journal]
    if len(set(ids)) != len(ids) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Duplicate or unordered command provenance')
    for index, command in enumerate(journal):
        target = np.asarray(command['target'], float)
        if target.shape != (7,) or not np.isfinite(target).all() or not 0 <= target[6] <= 100:
            raise ValueError('Invalid command target')
        if index == 0:
            continue
        row = commands.get(command['sequence'])
        if row is None:
            # An uncertain command must never enter a selected training window.
            continue
        if row['command'] != command or completion_clock(row, basis) < times[index]:
            raise ValueError('Wire command/journal or completion timing mismatch')
        previous = np.asarray(journal[index - 1]['target'], float)
        if row['method'] == 'MoveL':
            if command.get('kind') != 'arm' or target[6] != previous[6]:
                raise ValueError('MoveL must retain the preceding gripper target')
        elif command.get('kind') != 'gripper' or not np.array_equal(target[:6], previous[:6]):
            raise ValueError('MoveGripper must retain the preceding arm target')
    if not set(commands).issubset(ids):
        raise ValueError('Completed command absent from journal')


def check_frame_interval(previous, current):
    if previous is not None and not 0 < current-previous <= MAX_SOURCE_GAP_S:
        raise ValueError('Source frame gap or ordering is incompatible with a clip')


def grid_tolerance(maximum_error_s):
    if (isinstance(maximum_error_s,bool) or not isinstance(maximum_error_s,(int,float))
            or not np.isfinite(maximum_error_s) or not 0 <= maximum_error_s <= MAX_GRID_ERROR_S):
        raise ValueError('Grid tolerance must be below half the nominal frame period')
    return float(maximum_error_s)


def check_grid_time(origin, index, current, *, maximum_error_s=MAX_GRID_ERROR_S):
    tolerance=grid_tolerance(maximum_error_s)
    if abs(current-(origin+index/FPS)) > tolerance+1e-9:
        raise ValueError('Source frame is too far from its nominal 6 fps grid slot')


def resample_six_hz(entries, basis, *, maximum_error_s=MAX_GRID_ERROR_S):
    """Select nearest distinct real frames in disjoint nominal-time cells.

    Keep each observation/issued-target pair intact. Missing cells, source gaps,
    recording boundaries and known command phase changes start fresh clips.
    """
    tolerance=grid_tolerance(maximum_error_s)
    if not entries:return []
    groups=[];group=[];previous=None;previous_group=None
    for entry in entries:
        row,command=entry;t,clock=camera_clock(row['camera'])
        if clock!=basis:raise ValueError('Source frame clock domain changed')
        segment=(row.get('composition_source') or {}).get('segment')
        phase=(command.get('teacher') or {}).get('phase')
        current_group=(segment,phase)
        if previous is not None and (t<=previous or t-previous>MAX_SOURCE_GAP_S or current_group!=previous_group):
            if t<=previous:raise ValueError('Source frame order changed during sampling')
            if group:groups.append(group)
            group=[]
        group.append(entry);previous=t;previous_group=current_group
    if group:groups.append(group)
    clips=[]
    for group in groups:
        times=[camera_clock(row['camera'])[0] for row,_ in group]
        start=0
        while start<len(group):
            selected=[group[start]];last=start;origin=times[start]
            while last+1<len(group):
                target=origin+len(selected)/FPS
                right=bisect.bisect_left(times,target,lo=last+1)
                choices=[i for i in (right-1,right) if last<i<len(group)]
                if not choices:break
                index=min(choices,key=lambda i:(abs(times[i]-target),i))
                if (abs(times[index]-target)>tolerance+1e-9
                        or times[index]-times[last]>MAX_SOURCE_GAP_S):break
                selected.append(group[index]);last=index
            if len(selected)>=2:clips.append(selected)
            start=last+1
    return clips


def phase_target(journal, commands, timestamp, basis, phase):
    command = next((c for c in reversed(journal) if command_clock(c,basis) <= timestamp+1/FPS), None)
    if command is None:
        raise ValueError('No command provenance at the frame horizon')
    if command['sequence'] == journal[0]['sequence']:
        return None  # Initial observed hold was not sent as a physical command.
    record = commands.get(command['sequence'])
    if record is None:
        raise ValueError('Uncertain command entered a training window')
    return command if record['phase'] == phase else None


def curate(episode, output, *, original_episode=None):
    episode, output = Path(episode).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    review,review_sha = verify_success_review(episode, original_episode=original_episode)
    summary = json.loads((episode / 'summary.json').read_text())
    scope = json.loads((episode / 'scope.json').read_text())
    components = physical_components(episode)
    composite = scope.get('schema') == 'fr5-composite-logical-scope-v1'
    split_role = scope.get('split_role', 'training_pilot_trial')
    if split_role not in ('training_pilot_trial', 'training_complete_trial', 'held_out_complete_trial'):
        raise ValueError('Unknown source split reservation')
    phases_ordered = phase_order(scope)
    commands = validated_commands(episode)
    if any(row['phase'] not in phases_ordered for row in commands.values()):
        raise ValueError('Command phase is absent from the declared task schema')
    rows = [json.loads(s) for s in (episode / "lerobot/meta/capture.jsonl").read_text().splitlines()]
    if not rows or len({r['frame_index'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate source frame indices')
    basis = camera_clock(rows[0]['camera'])[1]
    if any(camera_clock(r['camera'])[1] != basis for r in rows):
        raise ValueError('Mixed source clock domains')
    row_times=[camera_clock(r['camera'])[0] for r in rows]
    if any(b<=a for a,b in zip(row_times,row_times[1:])):
        raise ValueError('Source frames are not strictly time ordered')
    image_rows = list(map(json.loads, (episode / "continuous/capture.jsonl").read_text().splitlines()))
    if any(camera_clock(r['camera'])[1] != basis for r in image_rows):
        raise ValueError('Image and action capture clock domains differ')
    images = {round(camera_clock(r['camera'])[0] * 1e6): r for r in image_rows}
    if len(images) != len(image_rows):
        raise ValueError('Duplicate source image timestamps')
    image_times=[camera_clock(r['camera'])[0] for r in image_rows]
    if any(b<=a for a,b in zip(image_times,image_times[1:])):
        raise ValueError('Source images are not strictly time ordered')
    image_hashes=composite_frame_provenance(episode,rows,image_rows)
    journal = json.loads((episode / "target-command-journal.json").read_text())
    validate_journal(journal, commands, basis)
    windows = []
    for row in commands.values():
        start, end, phase = command_clock(row['command'],basis)-.35, completion_clock(row,basis)+.35, row['phase']
        if windows and windows[-1]["phase"] == phase and start <= windows[-1]["end"]:
            windows[-1]["end"] = max(windows[-1]["end"], end)
        else:
            windows.append({"start": start, "end": end, "phase": phase})
    features = {
        "observation.state": {"dtype": "float32", "shape": (8+len(phases_ordered),),
                              "names": STATE_NAMES + ["gripper_measurement_valid"] + [f"phase_{p}" for p in phases_ordered]},
        "observation.images.front": {"dtype": "image", "shape": (360,640,3), "names": ["height","width","channels"]},
        "action": {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        "observation.source_frame_index": {"dtype": "int64", "shape": (1,), "names": ["raw_frame_index"]},
        "observation.camera_time_us": {"dtype": "int64", "shape": (1,), "names": ["camera_host_unix_us"]},
        "action.command_sequence": {"dtype": "int64", "shape": (1,), "names": ["submitted_command_sequence"]},
    }
    ds = LeRobotDataset.create(repo_id="local/fr5_teacher_pilot", root=output, fps=FPS,
                              robot_type="fairino_fr5_phase_conditioned_pilot", features=features,
                              use_videos=False, image_writer_threads=2)
    mapping, clips = [], []
    phase_boundary_frames_excluded = 0
    resampling_frames_excluded = 0
    try:
        for window in windows:
            selected=[]
            for row in rows:
                t=camera_clock(row['camera'])[0]
                if not window['start'] <= t <= window['end']:
                    continue
                cmd=phase_target(journal,commands,t,basis,window['phase'])
                if cmd is None:
                    phase_boundary_frames_excluded += 1
                else:
                    selected.append((row,cmd))
            sampled_clips=resample_six_hz(selected,basis)
            if not sampled_clips:
                raise ValueError("Too few frames for a completed command window")
            resampling_frames_excluded += len(selected)-sum(len(part) for part in sampled_clips)
            for sampled in sampled_clips:
                clip = {**window, "first_output_frame": len(mapping), "frames": 0,
                        "source_grid_origin":camera_clock(sampled[0][0]['camera'])[0],
                        "source_segment":sampled[0][0].get('composition_source',{}).get('segment')}
                previous = None
                for row,cmd in sampled:
                    t = camera_clock(row['camera'])[0]
                    check_frame_interval(previous, t)
                    check_grid_time(clip['source_grid_origin'],clip['frames'],t)
                    previous = t
                    # The target is a submitted command, never a future measured state.
                    state = row["robot"]
                    valid = gripper_feedback_valid(state)
                    measured = [*state["joints"], state["gripper_position"] if valid else 0., float(valid)]
                    measured += [float(window["phase"] == phase) for phase in phases_ordered]
                    image_row = images[round(t * 1e6)]
                    if image_row['camera'] != row['camera'] or image_row['robot'] != row['robot']:
                        raise ValueError('Image and action capture observations differ')
                    rgb_path=episode / "continuous" / image_row["rgb_file"]
                    if image_hashes is not None and file_sha256(rgb_path)!=image_hashes[image_row['frame_index']]:
                        raise ValueError('Composite RGB differs from original frame manifest')
                    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
                    if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
                        raise ValueError('Source front RGB must be 640x360 uint8')
                    ds.add_frame({"observation.state": np.array(measured,np.float32),
                        "observation.images.front": rgb, "action": np.array(cmd["target"],np.float32),
                        "observation.source_frame_index": np.array([row["frame_index"]],np.int64),
                        "observation.camera_time_us": np.array([round(row['camera']['source_timestamp_unix']*1e6)],np.int64),
                        "action.command_sequence": np.array([cmd["sequence"]],np.int64),
                        "task": TASKS[window["phase"]]})
                    mapping.append({"output_frame": len(mapping), "clip": len(clips), "phase": window["phase"],
                        "source_frame": row["frame_index"], "camera_time":row['camera']['source_timestamp_unix'],
                        "timeline_time":t,"timing_clock":basis,"target_command_sequence": cmd["sequence"],
                        "target_issued_timeline_time":command_clock(cmd,basis),
                        "target_issued_unix": cmd["issued_unix"], "gripper_valid": valid,
                        "composition_source": row.get('composition_source'),
                        "command_composition_source": commands.get(cmd['sequence'],{}).get('composition_source')})
                    clip["frames"] += 1
                ds.save_episode()
                clips.append(clip)
    finally:
        ds.finalize()
        ds.stop_image_writer()
    excluded_commands = []
    for file in episode.glob('command-*.json'):
        row=json.loads(file.read_text())
        if row.get('status')!='completed' or row.get('outcome',{}).get('return_code')!=0:
            excluded_commands.append(file.name)
    phases = sorted({c['phase'] for c in clips})
    complete = summary['outcome'] == 'pending_review' and set(phases) == set(phases_ordered) and not excluded_commands
    report = {"source_episode": str(episode), "source_original_episode": str(original_episode) if original_episode else None,
        "source_outcome_unchanged": summary["outcome"],
        "source_summary_sha256": hashlib.sha256((episode / "summary.json").read_bytes()).hexdigest(),
        "user_review_sha256": review_sha,
        "user_confirmed_success": review['verdict'] == 'success',
        "independent_physical_trials": 1, "temporal_clips": clips, "frames": len(mapping),
        "completed_commands": len(commands), "validated_targets_match_wire_arguments": True,
        "split_role": split_role,
        "training_use": "forbidden_reserved_evaluation" if split_role.startswith('held_out') else "pipeline_pilot_only",
        "deployment_ready": False, "is_held_out_trial": split_role.startswith('held_out'),
        "phases_present": phases,
        "complete_physical_episode": complete and not composite,
        "complete_logical_episode": complete,
        "source_is_composite": composite, "source_components": components,
        "capture_is_continuous_across_segments": False if composite else None,
        "phase_order": phases_ordered,
        "action_alignment": "Latest issued target at camera time +1/6s in timing_clock, restricted to the matching commanded phase; Unix retained as audit data",
        "phase_boundary_frames_excluded": phase_boundary_frames_excluded,
        "resampling_frames_excluded":resampling_frames_excluded,
        "nominal_fps": FPS, "maximum_grid_time_error_s": MAX_GRID_ERROR_S,
        "grid_boundary_epsilon_s":GRID_BOUNDARY_EPSILON_S,
        "maximum_source_gap_s":MAX_SOURCE_GAP_S,
        "phase_chunk_coverage":{
            phase:{'frames':sum(c['frames'] for c in clips if c['phase']==phase),
                   'clips':sum(c['phase']==phase for c in clips),
                   'clips_at_least_8_frames':sum(c['phase']==phase and c['frames']>=8 for c in clips),
                   'valid_eight_frame_chunks':sum(max(0,c['frames']-7) for c in clips if c['phase']==phase)}
            for phase in phases_ordered},
        "resampling":"Nearest distinct source frames on a 6Hz grid; split at missing slots, gaps or recording segments; source timestamps retained",
        "timing_clock":basis,
        "state_definition": f"6 measured joint degrees; measured gripper percent or0 if invalid; validity flag;{len(phases_ordered)} requested phase one-hot channels in phase_order",
        "state_masking": "Invalid gripper0 is a sentinel with validity0, never asserted as actual closed position",
        "phase_dependency": "Inference requires the high-level supervisor to supply the requested phase",
        "excluded_commands": excluded_commands,
        "excluded": "Long supervisor pauses and unsuccessful/uncertain command windows",
        "coverage_limit": ("Varied-phase schema; actual start displacement and approach require separate command and measurement evidence; fixed cup geometry and no random placement"
                           if phases_ordered == VARIED_PHASES else
                           "Same prealigned cup instance; no random placement or fresh approach examples")}
    (output / "meta/pilot.json").write_text(json.dumps(report,indent=2))
    (output / "meta/source_frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in mapping))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--original-episode",type=Path,help="Original absolute episode path if evidence paths predate relocation")
    args=p.parse_args()
    print(json.dumps(curate(args.episode,args.output,original_episode=args.original_episode),indent=2))


if __name__ == "__main__":
    main()
