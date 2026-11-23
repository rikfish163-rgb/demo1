"""One entry point for FR5 collect/collect-object, data4 training and pi0.5 run.

Motion occurs only in collect/collect-object/run with --execute. Reviews contain the user's
actual whole-episode verdict. Publishing code, exporting data and training do
not access robot actuators. All large artifacts remain on server4090 data4.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCAL_STATE = Path.home()/'.local/state/fr5-system'
SPOOL = Path.home()/'.local/state/fr5-recording-spool'
REMOTE = '/home/data4/hts/demo1/fr5-learning'
PYTHON = REMOTE+'/.venv/bin/python'
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z')


def simple_name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError('Use a simple episode/dataset/run name, without path components')
    return value


def generated(prefix):
    return prefix+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]


def configuration():
    path = LOCAL_STATE/'config.json'
    return json.loads(path.read_text()) if path.exists() else {}


def configure(**values):
    LOCAL_STATE.mkdir(parents=True, exist_ok=True)
    config = configuration(); config.update(values)
    path = LOCAL_STATE/'config.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)
    return config


def ssh_args(command):
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
            '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2', '4090', command]


def remote_json(code, request, *, timeout=30):
    result = subprocess.run(ssh_args(shlex.join([PYTHON, '-c', code])),
        input=json.dumps(request, ensure_ascii=False), text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:] or 'Remote operation failed')
    return json.loads(result.stdout)


@contextmanager
def robot_lease():
    """One system process owns J1; no per-action user confirmation."""
    LOCAL_STATE.mkdir(parents=True, exist_ok=True)
    with (LOCAL_STATE/'j1.lock').open('a+') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another fr5-system process is already operating J1') from exc
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def status():
    report = {'configuration': configuration(), 'robot_commands_sent': 0}
    try:
        report['preview'] = json.load(urllib.request.urlopen('http://127.0.0.1:8765/status', timeout=3))
    except Exception as exc:
        report['preview'] = {'available': False, 'error': str(exc)}
    report['server'] = remote_json(r'''
import json,hashlib,subprocess,os
from pathlib import Path
q=json.load(__import__('sys').stdin);root=Path(q['root'])
disk=os.statvfs(root);sources=[];pending=[];failed=[]
for scope_path in sorted((root/'recording-spool').glob('*/materialized/scope.json')):
 p=scope_path.parent
 try:
  scope=json.loads(scope_path.read_text());summary=json.loads((p/'summary.json').read_text())
  if not scope.get('whole_task'):continue
  name=p.parent.name
  review_path=p/'user-review.json'
  review=json.loads(review_path.read_text()) if review_path.exists() else {}
  if (review.get('verdict')=='success' and review.get('summary_sha256')==hashlib.sha256((p/'summary.json').read_bytes()).hexdigest() and review.get('scope_sha256')==hashlib.sha256(scope_path.read_bytes()).hexdigest()):sources.append(name)
  elif summary.get('outcome')=='pending_review' and not review:pending.append(name)
  else:failed.append(name)
 except (OSError,ValueError):continue
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
checkpoints=[p.parent.name for p in sorted((root/'checkpoints').glob('*/manifest.json'))]
print(json.dumps({'data4_free_bytes':disk.f_bavail*disk.f_frsize,'successful_whole_task_sources':sources,'pending_whole_task_sources':pending,'failed_or_interrupted_sources':failed,'packaged_checkpoints':checkpoints,'gpu':gpu.stdout.strip()}))
''', {'root':REMOTE})
    return report


def publish(name=None):
    """Send a small immutable source snapshot; never overwrite server datasets."""
    name = simple_name(name or generated('fr5-system'))
    remote = REMOTE+'/src/'+name
    paths = sorted((ROOT/'fr5_runtime').rglob('*.py'))
    paths += sorted((ROOT/'fr5_runtime/web').glob('*.html'))
    paths += [ROOT/'run-collection-console.sh']
    paths += sorted((ROOT/'fr5_lerobot/src/lerobot_robot_fairino/web').glob('*.html'))
    paths += sorted((ROOT/'fr5_lerobot/src').rglob('*.py'))
    paths += sorted((ROOT/'fr5_lerobot/training').glob('*.py'))
    paths += sorted((ROOT/'fr5_lerobot/diagnostics').glob('*.py'))
    paths += sorted((ROOT/'fr5_lerobot/tests').glob('*.py'))
    paths += sorted((ROOT/'fr5_jev').glob('*.py'))
    paths += sorted((ROOT/'fr5_jev/decisions/cases').glob('*.json'))
    paths += sorted(path for path in (ROOT/'hardware').rglob('*') if path.is_file())
    metadata = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
        raw = json.dumps(metadata, sort_keys=True, indent=2).encode()
        info = tarfile.TarInfo('source-manifest.json'); info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    command = 'mkdir '+shlex.quote(remote)+' && tar -xz -C '+shlex.quote(remote)
    subprocess.run(ssh_args(command), input=buffer.getvalue(), check=True, timeout=60)
    proof = remote_json(r'''
import sys,json,hashlib
from pathlib import Path
p=Path(json.load(sys.stdin)['source']);manifest=json.loads((p/'source-manifest.json').read_text())
for name,digest in manifest.items():
 if hashlib.sha256((p/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source readback mismatch: '+name)
print(json.dumps({'source':str(p),'files':len(manifest),'source_manifest_sha256':hashlib.sha256((p/'source-manifest.json').read_bytes()).hexdigest()}))
''', {'source':remote})
    configure(server_source=remote, source_manifest_sha256=proof['source_manifest_sha256'])
    return proof


def server_source():
    source = configuration().get('server_source')
    if not source or not source.startswith(REMOTE+'/src/'):
        raise RuntimeError('Run ./fr5-system publish once to install the current server code')
    return source


def server_program(arguments):
    source = server_source()
    command = shlex.join(['env', 'PYTHONPATH='+source+'/fr5_lerobot/src:'+source,
                         'TMPDIR='+REMOTE+'/tmp', PYTHON, *arguments])
    result = subprocess.run(ssh_args(command))
    if result.returncode:
        raise RuntimeError(f'Server command exited {result.returncode}')


def checkpoint_spec(checkpoint):
    """Read the selected bundle's identity and execution settings together."""
    from lerobot_robot_fairino.whole_task_policy import data4_path
    spec=remote_json('''
import json,sys,hashlib
from pathlib import Path
p=Path(json.load(sys.stdin)['checkpoint'])
manifest=p/'stream-manifest.json'
raw=manifest.read_bytes();m=json.loads(raw);kind=m.get('policy_contract')
result={'policy_contract':kind,'checkpoint_sha256':hashlib.sha256(raw).hexdigest()}
if isinstance(kind,dict) and kind.get('name')=='fr5-joint-position-stream-v1':
 config=json.loads((p/'config.json').read_text());n=config.get('n_action_steps',50)
 if type(n) is not int or not 1<=n<=50:raise ValueError('Invalid sealed n_action_steps')
 if 'n_action_steps' in m and (type(m['n_action_steps']) is not int or m['n_action_steps']!=n):
  raise ValueError('Manifest and config n_action_steps differ')
 result['n_action_steps']=n
 result['observation_context_profile']=m.get('observation_context_profile','none')
print(json.dumps(result))
''',
        {'checkpoint': data4_path(checkpoint)})
    value=spec.get('policy_contract')
    if not isinstance(value,dict) or value.get('name')!='fr5-joint-position-stream-v1':
        raise ValueError('Expected a continuous FR5 pi0.5 checkpoint')
    if isinstance(value,dict) and isinstance(value.get('name'),str):
        known=configuration().get('checkpoint_contracts',{})
        known=known if isinstance(known,dict) else {}
        configure(checkpoint_contracts={**known,data4_path(checkpoint):value})
    return spec


def checkpoint_contract(checkpoint):
    return checkpoint_spec(checkpoint)['policy_contract']


def archive_stream(name, *, render=False):
    """Finish evidence after J1 is released; this cannot send robot commands."""
    from fr5_jev.stream_storage import StreamStorage
    name=simple_name(name)
    root=SPOOL/name
    outbox=root/'frame-outbox'
    finalized=outbox/'archive-finalized.json'
    manifest=root/'control-manifest.json'
    scope_path=root/'scope.json'
    scope=json.loads(scope_path.read_text()) if scope_path.is_file() else {}
    began=time.monotonic()
    print(json.dumps({'status':'archiving','episode':name,'robot_lease_held':False}),flush=True)
    while not finalized.exists():
        # Evaluation closure precedes the final summary and control snapshot.
        # Whole-task demonstrations do not write the evaluation-only marker.
        if (root/'stream-closed.json').exists() and (
                manifest.exists() or scope.get('dataset_role')!='policy_evaluation'):
            break
        error=outbox/'archive-error.json'
        if error.exists():
            raise RuntimeError('Archive closure failed: '+error.read_text())
        if time.monotonic()-began>180:
            raise TimeoutError('Archive still pending; use archive-stream to resume without robot movement')
        time.sleep(.1)
    storage=StreamStorage.from_closed(root)
    if not finalized.exists():
        state=remote_json('''
import json,sys
from pathlib import Path
root=Path(json.load(sys.stdin)['root'])
print(json.dumps({'materialized':(root/'materialized').exists(),
                  'control_manifest':(root/'control-manifest.json').exists()}))
''', {'root':str(storage.remote_root)})
        # Existing materialization is verified in place, including records
        # whose local checkpoint caches have already been released.
        if not state['materialized'] and (
                not manifest.exists() or not state['control_manifest']):
            storage.finalize_control()
    proof=storage.materialize(timeout=600)
    receipt=outbox/'materialization-receipt.json'
    receipt.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    result={'materialization':proof}
    if render:
        from fr5_runtime import console_data as fr5_console_data
        result['replay']=fr5_console_data.render_replay(name)
    return result


def observe(name=None):
    from fr5_jev.observe import capture
    name = simple_name(name or generated('scene'))
    directory = SPOOL/name
    report = capture(directory)
    configure(last_observation=str(directory))
    return {'directory':str(directory), 'image':str(directory/'color.jpg'),
            'robot_state':report['robot_after'], 'robot_commands_sent':0}


def _collect(args):
    from fr5_jev.whole_task_collection import prepare
    name = simple_name(args.name or generated('cup'))
    if args.observation:
        observation = Path(args.observation).resolve()
    else:
        observation = Path(observe()['directory'])
    directory = SPOOL/name
    plan = prepare(directory, observation, args.cup_uv, grasp_reference=args.grasp_reference,
        speed=args.speed, split='held_out_complete_trial' if args.held_out else 'training_complete_trial',
        hold_seconds=args.hold_seconds, repetitions=args.repetitions, retreat_mode=args.retreat)
    if args.planner_note:
        from fr5_jev.whole_task_collection import record_llm_route_plan
        record_llm_route_plan(directory,args.planner_note)
    configure(last_collection=name)
    if args.execute:
        subprocess.run([sys.executable, '-m', 'fr5_jev.whole_task_collection', 'run',
                        '--directory', str(directory), '--execute'], cwd=ROOT, check=True)
        summary = json.loads((directory/'summary.json').read_text())
        return {'episode':name,'remote_episode':REMOTE+'/recording-spool/'+name+'/materialized',
                'outcome':summary['outcome'],'requires_whole_episode_user_verdict':True,
                'success_count_increment':0}
    return plan


def collect(args):
    with robot_lease():
        return _collect(args)


def review(name, verdict, statement, *, input_method='chat_message'):
    name = simple_name(name)
    source = server_source()
    return remote_json(r'''
import sys,json,hashlib,time
from pathlib import Path
q=json.load(sys.stdin);sys.path.insert(0,q['code']+'/fr5_lerobot/src')
from lerobot_robot_fairino.human_review import record_review
p=Path(q['episode']);scope=json.loads((p/'scope.json').read_text());summary=json.loads((p/'summary.json').read_text())
if scope.get('dataset_role')=='policy_evaluation' or q['verdict']=='failure':
 schema='fr5-policy-user-review-v1' if scope.get('dataset_role')=='policy_evaluation' else 'fr5-user-review-v1'
 review={'schema':schema,'verdict':q['verdict'],'user_statement':q['statement'],'reviewer':'user','source':'direct_user_message','input_method':q['input_method'],'summary_sha256':hashlib.sha256((p/'summary.json').read_bytes()).hexdigest(),'scope_sha256':hashlib.sha256((p/'scope.json').read_bytes()).hexdigest(),'reviewed_at_unix':time.time(),'training_ready':False}
 with (p/'user-review.json').open('x') as f:json.dump(review,f,ensure_ascii=False,indent=2)
else:
 review=record_review(p,verdict=q['verdict'],user_statement=q['statement'],input_method=q['input_method'])
print(json.dumps(review,ensure_ascii=False))
''', {'code':source,'episode':REMOTE+'/recording-spool/'+name+'/materialized',
      'verdict':verdict,'statement':statement,'input_method':input_method})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('status')
    pub = sub.add_parser('publish');pub.add_argument('--name')
    obs = sub.add_parser('observe');obs.add_argument('--name')
    coll = sub.add_parser('collect')
    coll.add_argument('--name');coll.add_argument('--observation',type=Path)
    coll.add_argument('--cup-uv',type=float,nargs=2,required=True)
    coll.add_argument('--grasp-reference',type=Path)
    coll.add_argument('--hold-seconds',type=float,default=5.)
    coll.add_argument('--repetitions',type=int,choices=(1,2),default=1)
    coll.add_argument('--speed',type=float,default=10.)
    coll.add_argument('--retreat',choices=('short','full'),default='short')
    coll.add_argument('--planner-note',help='Actual current-chat scene/route reasoning for an episode-local LLM plan')
    coll.add_argument('--held-out',action='store_true');coll.add_argument('--execute',action='store_true')
    object_coll = sub.add_parser('collect-object',help='Prepare or execute a whole-task object demonstration')
    object_coll.add_argument('--name',required=True)
    object_coll.add_argument('--object-description',required=True)
    selection = object_coll.add_mutually_exclusive_group(required=True)
    selection.add_argument('--rgb-pixel',type=float,nargs=2,metavar=('U','V'))
    selection.add_argument('--mask',type=Path,metavar='PNG')
    object_coll.add_argument('--support-rgb-pixel',type=float,nargs=2,metavar=('U','V'),
                             help='Pixel on the supporting surface for measured support geometry')
    object_coll.add_argument('--closing-percent',type=int,required=True)
    object_coll.add_argument('--grasp-offset',type=float,nargs=3,metavar=('X','Y','Z'),required=True)
    object_coll.add_argument('--approach',choices=('top-down','side','tilted'),default='tilted')
    object_coll.add_argument('--grasp-below-top-mm',type=float,default=10.)
    object_coll.add_argument('--side-retraction-mm',type=float,default=0.)
    object_coll.add_argument('--tilt-from-down-deg',type=float,default=55.)
    object_coll.add_argument('--tilted-closing-edge',choices=('long','short'),default='long')
    object_coll.add_argument('--split',choices=('training_complete_trial','held_out_complete_trial'),
                             default='training_complete_trial')
    object_coll.add_argument('--execute',action='store_true')
    rev = sub.add_parser('review')
    rev.add_argument('--episode',required=True);rev.add_argument('--verdict',choices=('success','failure'),required=True)
    rev.add_argument('--statement',required=True,help='The actual user verdict, never an inferred approval')
    stream_export = sub.add_parser('export-stream')
    stream_export.add_argument('--episodes',nargs='+',required=True)
    stream_export.add_argument('--dataset',required=True)
    stream_export.add_argument('--allow-recorded-initial-hold',action='store_true')
    stream_train = sub.add_parser('train-stream')
    stream_train.add_argument('--dataset',required=True)
    stream_train.add_argument('--run',required=True)
    stream_train.add_argument('--steps',type=int,required=True)
    stream_train.add_argument('--gpu',type=int)
    stream_train.add_argument('--rank',type=int,default=32)
    stream_train.add_argument('--n-action-steps',type=int,
        help='Checkpoint setting: new runs default to 50; parent runs inherit their sealed setting')
    stream_train.add_argument('--adaptation',choices=('lora','full-expert-language-lora','full-expert-vision-language-lora'),default='lora')
    stream_train.add_argument('--action-representation',
        choices=('absolute-joint-position-v1','joint-delta-from-anchor-v1','cup-relative-tcp-v1',
                 'cup-relative-tcp-delta-v1','cup-relative-tcp-tick-delta-v1'),
        default='absolute-joint-position-v1')
    stream_train.add_argument('--image-profile',choices=('full-v1','full-and-workspace-v1','workspace-only-v1','cup-centered-v1'),
        help='Image preprocessing profile; omitted values inherit the parent checkpoint')
    stream_train.add_argument('--sampling-profile',choices=('balanced-v2','approach-open-v1',
                                                            'early-switch-context-v1','cup-timing-contact-return-v1',
                                                            'cup-clock-hold-return-v1','cup-camera-clock-events-v1',
                                                            'cup-camera-clock-boundaries-v1','cup-camera-clock-support-v1',
                                                            'cup-camera-clock-coupled-support-v1',
                                                            'uniform-all-frames-v1',
                                                            'uniform-switch-v1','uniform-switch-hold-v1'),
        help='Training anchor sampling; omitted values inherit the parent profile')
    stream_train.add_argument('--observation-context-profile',choices=('none','state-history-v1',
        'state-history-physical-v1','state-history-physical-initial-cup-v1','state-history-physical-cup-tcp-v1',
        'state-history-physical-cup-tracking-v1','state-history-physical-cup-timing-v1',
        'state-history-physical-cup-action-clock-v1','state-history-physical-cup-aligned-v1',
        'state-history-physical-cup-setdown-v1','state-history-physical-cup-setdown-clean-v1'),
        help='Measured state history used to condition the policy')
    stream_train.add_argument('--cup-roll-state-min-span-rad',type=float,
        help='Minimum cup-relative state roll range; omitted values inherit the parent')
    stream_train.add_argument('--batch-size',type=int,default=1)
    stream_train.add_argument('--gradient-accumulation',type=int,default=4)
    stream_train.add_argument('--learning-rate',type=float,default=1e-5)
    stream_train.add_argument('--lr-schedule',choices=('constant','cosine'),default='constant')
    stream_train.add_argument('--decay-learning-rate',type=float)
    stream_train.add_argument('--first-action-weight',type=float,default=8.)
    stream_train.add_argument('--executed-prefix-weight',type=float,default=1.)
    stream_train.add_argument('--vertical-weight',type=float,default=1.)
    stream_train.add_argument('--gripper-weight',type=float,default=3.)
    stream_train.add_argument('--xyz-loss-mode',choices=('clean','terminal'))
    stream_train.add_argument('--save-every',type=int,default=500)
    stream_train.add_argument('--seed',type=int,help='Training and sampling seed')
    stream_parent=stream_train.add_mutually_exclusive_group()
    stream_parent.add_argument('--resume')
    stream_parent.add_argument('--initialize',help='Explicit weights/normalization warm start with a new optimizer')
    archive = sub.add_parser('archive-stream')
    archive.add_argument('--episode',required=True)
    archive.add_argument('--render',action='store_true')
    check = sub.add_parser('inspect');check.add_argument('--checkpoint',required=True)
    run = sub.add_parser('run')
    run.add_argument('--checkpoint',required=True);run.add_argument('--task',required=True)
    run.add_argument('--name');run.add_argument('--gpu',type=int)
    run.add_argument('--seconds',type=float,help='Optional explicitly requested duration; otherwise run until stopped')
    run.add_argument('--scene-json',type=Path,help='Measured table scene for joint-policy prefix preflight')
    run.add_argument('--prefix-steps',type=int,
        help='Observe again after this many model actions, at most the checkpoint prefix')
    run.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if args.operation=='status':result=status()
    elif args.operation=='publish':result=publish(args.name)
    elif args.operation=='observe':
        with robot_lease():result=observe(args.name)
    elif args.operation=='collect':result=collect(args)
    elif args.operation=='collect-object':
        from fr5_jev.collect_object import collect as collect_object
        result=collect_object(name=args.name,object_description=args.object_description,
            rgb_pixel=args.rgb_pixel,support_rgb_pixel=args.support_rgb_pixel,
            mask_path=args.mask,closing_percent=args.closing_percent,
            grasp_offset=args.grasp_offset,approach=args.approach,
            grasp_below_top_mm=args.grasp_below_top_mm,side_retraction_mm=args.side_retraction_mm,
            tilt_from_down_deg=args.tilt_from_down_deg,tilted_closing_edge=args.tilted_closing_edge,
            split=args.split,execute=args.execute)
    elif args.operation=='review':result=review(args.episode,args.verdict,args.statement)
    elif args.operation=='export-stream':
        dataset=simple_name(args.dataset)
        command=['-m','lerobot_robot_fairino.export_stream_policy','--output',REMOTE+'/datasets/'+dataset]
        for name in args.episodes:
            command += ['--source',REMOTE+'/recording-spool/'+simple_name(name)+'/materialized']
        if args.allow_recorded_initial_hold:
            command += ['--allow-recorded-initial-hold']
        server_program(command)
        configure(last_stream_dataset=dataset)
        result={'dataset':REMOTE+'/datasets/'+dataset,'action_dim':7}
    elif args.operation=='train-stream':
        dataset,run_name=simple_name(args.dataset),simple_name(args.run)
        command=[server_source()+'/fr5_lerobot/training/train_stream_pi05.py',
            '--dataset',REMOTE+'/datasets/'+dataset,'--run',REMOTE+'/checkpoints/'+run_name,
            '--steps',str(args.steps),'--rank',str(args.rank),'--batch-size',str(args.batch_size),
            '--gradient-accumulation',str(args.gradient_accumulation),
            '--adaptation',args.adaptation,'--action-representation',args.action_representation,
            '--learning-rate',str(args.learning_rate),'--first-action-weight',str(args.first_action_weight),
            '--gripper-weight',str(args.gripper_weight),
            '--lr-schedule',args.lr_schedule,
            '--save-every',str(args.save_every)]
        if args.executed_prefix_weight != 1.:
            command+=['--executed-prefix-weight',str(args.executed_prefix_weight)]
        if args.vertical_weight != 1.:
            command+=['--vertical-weight',str(args.vertical_weight)]
        if args.xyz_loss_mode is not None:
            command+=['--xyz-loss-mode',args.xyz_loss_mode]
        if args.decay_learning_rate is not None:
            command+=['--decay-learning-rate',str(args.decay_learning_rate)]
        if args.n_action_steps is not None:command+=['--n-action-steps',str(args.n_action_steps)]
        if args.image_profile is not None:command+=['--image-profile',args.image_profile]
        if args.sampling_profile is not None:command+=['--sampling-profile',args.sampling_profile]
        if args.observation_context_profile is not None:
            command+=['--observation-context-profile',args.observation_context_profile]
        if args.cup_roll_state_min_span_rad is not None:
            command+=['--cup-roll-state-min-span-rad',str(args.cup_roll_state_min_span_rad)]
        if args.gpu is not None:command+=['--gpu',str(args.gpu)]
        if args.seed is not None:command+=['--seed',str(args.seed)]
        if args.resume:
            from lerobot_robot_fairino.whole_task_policy import data4_path
            command+=['--resume',data4_path(args.resume)]
        if args.initialize:
            from lerobot_robot_fairino.whole_task_policy import data4_path
            command+=['--initialize',data4_path(args.initialize)]
        server_program(command)
        checkpoint=REMOTE+'/checkpoints/'+run_name+f'/step-{args.steps:06d}'
        # Training completion creates a candidate. Hardware task success is a
        # separate verified result and is never fabricated by this command.
        configure(last_stream_candidate=checkpoint)
        result={'checkpoint':checkpoint,'task_success_verified':False}
    elif args.operation=='archive-stream':
        result=archive_stream(args.episode,render=args.render)
    elif args.operation=='inspect':
        from lerobot_robot_fairino.whole_task_policy import data4_path
        checkpoint=data4_path(args.checkpoint)
        checkpoint_spec(checkpoint)
        subprocess.run(ssh_args(shlex.join([PYTHON,checkpoint+'/serve_stream_pi05.py',
            '--checkpoint',checkpoint,'--inspect'])),check=True)
        result={'status':'inspected','checkpoint':checkpoint,'robot_commands_sent':0}
    else:
        selected=checkpoint_spec(args.checkpoint)
        from lerobot_robot_fairino.stream_runtime import run as run_policy
        options={'seconds':args.seconds,'expected_action_steps':selected['n_action_steps'],
                 'expected_sha':selected['checkpoint_sha256']}
        if args.scene_json is not None:
            options['joint_scene']=json.loads(args.scene_json.read_text(encoding='utf-8'))
        if args.prefix_steps is not None:
            options['execution_steps']=args.prefix_steps
        name=simple_name(args.name or generated('pi05'))
        claimed=False
        run_error=None
        result={}
        try:
            with robot_lease():
                claimed=True
                result=run_policy(args.checkpoint,None,args.task,SPOOL/name,execute=args.execute,gpu=args.gpu,
                    **options)
        except BaseException as exc:
            run_error=exc
            result={'status':'policy_failed','error':f'{type(exc).__name__}: {exc}',
                    'episode':name,'task_success_verified':False}
        finally:
            if args.execute and claimed and LOCAL_STATE.is_dir():
                # Keep ownership outside the frozen evidence tree. Background
                # archiving must not reserve J1 after this process releases it.
                released=LOCAL_STATE/('released-'+name+'.json')
                released.write_text(json.dumps({'pid':os.getpid(),'episode':name,
                    'robot_lease_released':True,'time_unix':time.time()})+'\n')
        if claimed and (SPOOL/name).is_dir():
            configure(last_policy_episode=name)
        if args.execute and claimed and (SPOOL/name/'summary.json').is_file():
            try:
                result['archive']=archive_stream(name,render=True)
            except Exception as exc:
                result['archive_error']=str(exc)
        if run_error is not None:
            print(json.dumps(result,ensure_ascii=False),flush=True)
            raise run_error
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
