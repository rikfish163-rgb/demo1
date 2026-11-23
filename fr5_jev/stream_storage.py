"""Small-file control channel for a fresh asynchronous 4090 recording spool.

Frames use a separate long-lived binary stream. This channel confirms each
scope/command journal write after server fsync and is never called under a
capture or dispatch lock. Importing this module opens no connection.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import threading
import time
from types import SimpleNamespace

from .async_spool import AsyncFrameSpool
from .durable_storage import NAME, REMOTE_BASE, REMOTE_HOST
from . import stream_protocol

MAX_FILE = 16 * 1024 * 1024
MAX_BATCH = 32 * 1024 * 1024
MAX_BATCH_FILES = 32
MAX_LOCAL_CONTROL = 128 * 1024 * 1024
# Preserve a short baseline for small command records while allowing larger
# finalization batches to cross the measured ~1 MiB/s link. The conservative
# rate and hard cap keep a stalled control transfer bounded.
CONTROL_TIMEOUT_BASE_S = 15.0
CONTROL_TIMEOUT_RATE_BPS = 512 * 1024
CONTROL_TIMEOUT_PER_FILE_S = 1.0
CONTROL_TIMEOUT_MAX_S = 90.0
LOCAL_STREAM_BASE = Path.home() / ".local/state/fr5-recording-spool"
CODEC = (Path(__file__).resolve().parent.parent / "fr5_lerobot/src/lerobot_robot_fairino/frame_codec.py")
MATERIALIZER = Path(__file__).resolve().parent / "materialize_stream.py"


def _control_timeout(payload_bytes, file_count=0):
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise ValueError("Control payload size must be a nonnegative integer")
    if type(file_count) is not int or file_count < 0:
        raise ValueError("Control file count must be a nonnegative integer")
    return min(CONTROL_TIMEOUT_MAX_S, CONTROL_TIMEOUT_BASE_S
               + payload_bytes / CONTROL_TIMEOUT_RATE_BPS + file_count * CONTROL_TIMEOUT_PER_FILE_S)


_REMOTE_PUT = r'''
import hashlib,json,os,stat,struct,sys,uuid
from pathlib import Path
def read_exact(n):
 b=bytearray()
 while len(b)<n:
  part=sys.stdin.buffer.read(n-len(b))
  if not part:raise EOFError('truncated control payload')
  b.extend(part)
 return bytes(b)
def sync_dir(p):
 fd=os.open(p,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
 try:os.fsync(fd)
 finally:os.close(fd)
size=struct.unpack('!I',read_exact(4))[0]
if not 0<size<=1048576:raise ValueError('bad request')
q=json.loads(read_exact(size))
base=Path('/home/data4/hts/demo1/fr5-learning/recording-spool')
root=base/q['name']
if root.parent!=base or root.resolve()!=root or root.is_symlink() or not root.is_dir():
 raise ValueError('not a fresh spool root')
marker=json.loads((root/'INCOMPLETE').read_text())
if marker.get('session')!=q['session']:raise ValueError('session mismatch')
free=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
if free<q['required_bytes']:raise OSError('insufficient remote storage')
files=q['files']
if not isinstance(files,list) or sum(x['size'] for x in files)>33554432:raise ValueError('bad batch')
seen=set();verified={}
for item in files:
 name=item['path'];n=item['size'];expected=item['sha256']
 p=Path(name)
 if (p.is_absolute() or not p.parts or '..' in p.parts or name in seen
     or n<0 or n>16777216 or len(expected)!=64):raise ValueError('bad file')
 seen.add(name)
 parent=root
 for part in p.parts[:-1]:
  parent=parent/part
  if not parent.exists():
   parent.mkdir(mode=0o700);sync_dir(parent.parent)
  if parent.is_symlink() or not parent.is_dir() or parent.resolve()!=parent:
   raise ValueError('unsafe directory')
 target=parent/p.name
 if target.is_symlink() or (target.exists() and not target.is_file()):raise ValueError('unsafe file')
 if (root/'control-manifest.json').exists() and target.exists():
  if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
   raise ValueError('closed control snapshot conflicts with requested file')
 temp=parent/('.control-'+uuid.uuid4().hex)
 digest=hashlib.sha256()
 try:
  with temp.open('xb') as out:
   left=n
   while left:
    block=read_exact(min(left,65536));out.write(block);digest.update(block);left-=len(block)
   out.flush();os.fsync(out.fileno())
  if digest.hexdigest()!=expected:raise ValueError('control hash mismatch')
  os.replace(temp,target);sync_dir(parent)
  disk=hashlib.sha256()
  with target.open('rb') as verify:
   for block in iter(lambda:verify.read(65536),b''):disk.update(block)
  if disk.hexdigest()!=expected:raise ValueError('server readback hash mismatch')
 finally:
  if temp.exists():temp.unlink()
 verified[name]=digest.hexdigest()
print(json.dumps({'session':q['session'],'files':verified,'free_bytes':free}),flush=True)
'''


def new_stream_recording_path(name):
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError("Recording name must be a simple filename")
    LOCAL_STREAM_BASE.mkdir(mode=0o700, parents=True, exist_ok=True)
    if LOCAL_STREAM_BASE.is_symlink() or LOCAL_STREAM_BASE.resolve() != LOCAL_STREAM_BASE:
        raise RuntimeError("Local outbox parent must not contain symlinks")
    path = LOCAL_STREAM_BASE / name
    if path.exists() or path.is_symlink():
        raise FileExistsError("Choose a fresh recording name")
    return path


class StreamStorage:
    """Fresh stream with explicitly limited local unacknowledged data."""

    def __init__(self, root, scope_sha, frame_spool, *, required_bytes):
        self.root = Path(root)
        self.scope_sha = scope_sha
        self.frame_spool = frame_spool
        self.required_bytes = required_bytes
        self.remote_root = REMOTE_BASE / self.root.name
        self.codec_sha256 = hashlib.sha256(CODEC.read_bytes()).hexdigest()
        self.protocol_sha256 = hashlib.sha256(Path(stream_protocol.__file__).read_bytes()).hexdigest()
        self._sync_lock = threading.Lock()

    @classmethod
    def start_readonly(cls, root, scope_sha, *, required_bytes, process_factory=None, spool_settings=None):
        """Start a diagnostic only; this method does not authorize motion."""
        root = Path(root)
        if (root.parent != LOCAL_STREAM_BASE or not NAME.fullmatch(root.name)
                or root.is_symlink() or not root.is_dir() or root.resolve() != root):
            raise ValueError("Fresh local stream root required")
        if not isinstance(required_bytes, int) or isinstance(required_bytes, bool) or required_bytes <= 0:
            raise ValueError("A positive remote capacity budget is required")
        scope_bytes = (root / "scope.json").read_bytes()
        if hashlib.sha256(scope_bytes).hexdigest() != scope_sha:
            raise ValueError("Scope hash mismatch")
        scope = json.loads(scope_bytes)
        if not isinstance(scope, dict):
            raise ValueError("Scope must be an object")
        for key in ("geometry_path", "manual_action_plan_path", "lateral_geometry_path"):
            if key in scope:
                evidence = Path(scope[key])
                if (not evidence.is_absolute() or evidence.resolve() != evidence
                        or not evidence.is_relative_to(root) or not evidence.is_file()):
                    raise ValueError(f"{key} must be a regular file inside the new session")
        for item in scope.get("evidence", []):
            if isinstance(item, str) and Path(item).is_absolute():
                evidence = Path(item)
                if (evidence.resolve() != evidence or not evidence.is_relative_to(root) or not evidence.is_file()):
                    raise ValueError("Absolute evidence must be inside the new session")
        if any((root / name).exists() for name in ("frame-outbox", "SESSION_STARTED", "continuous", "lerobot")):
            raise ValueError("Recording root already claimed")
        if (root.stat().st_dev == os.stat('/dev/shm').st_dev):
            raise ValueError("The outbox must survive reboot")
        kwargs = {} if process_factory is None else {"process_factory": process_factory}
        if spool_settings is not None:
            if set(spool_settings) - {'max_frames','max_bytes','max_unacked_age'}:
                raise ValueError('Unknown frame buffer setting')
            kwargs.update(spool_settings)
        protocol_bytes = Path(stream_protocol.__file__).read_bytes()
        codec_bytes = CODEC.read_bytes()
        spool = AsyncFrameSpool(root, root.name, **kwargs).start()
        try:
            storage = cls(root, scope_sha, spool, required_bytes=required_bytes)
            if (storage.codec_sha256 != hashlib.sha256(codec_bytes).hexdigest()
                    or storage.protocol_sha256 != hashlib.sha256(protocol_bytes).hexdigest()):
                raise RuntimeError("Recorder code changed during stream startup")
            files = []
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Prepared evidence must not include symlinks")
                if path.is_file() and not path.is_relative_to(spool.outbox):
                    files.append(path)
            storage.sync_files(*files)
            storage._push({"wire/frame_codec.py": codec_bytes,
                           "wire/stream_protocol.py": protocol_bytes,
                           "wire/materialize_stream.py": MATERIALIZER.read_bytes()})
            storage.ensure_local_capacity(reserve_bytes=24 * 1024**2)
            return storage
        except BaseException:
            spool.abort()
            raise

    @classmethod
    def start_for_episode(cls, root, scope_sha, *, required_bytes):
        """Require current test/diagnostic proof, then claim a fresh server spool."""
        from .stream_readiness import validate_readiness
        validate_readiness()
        return cls.start_readonly(root, scope_sha, required_bytes=required_bytes)

    @classmethod
    def from_closed(cls, root):
        """Reopen only the file channel of an acknowledged closed recording."""
        root = Path(root)
        if (root.parent != LOCAL_STREAM_BASE or not NAME.fullmatch(root.name)
                or root.is_symlink() or not root.is_dir() or root.resolve() != root):
            raise ValueError('A closed local recording directory is required')
        def read(name):
            path = root/name
            if path.is_symlink() or path.resolve() != path or not path.is_file():
                raise ValueError('Invalid closed recording metadata')
            return json.loads(path.read_text())
        closed = read('stream-closed.json')
        scope = read('scope.json')
        scope_sha = hashlib.sha256((root/'scope.json').read_bytes()).hexdigest()
        # Early read-only diagnostic captures had no control manifest or task
        # summary. They can only be reopened to verify an existing archive.
        legacy_diagnostic = (scope.get('diagnostic_only') is True
            and scope.get('training_ready') is False
            and type(scope.get('physical_commands_sent')) is int and scope['physical_commands_sent'] == 0
            and not (root/'summary.json').exists() and not (root/'summary.json').is_symlink()
            and not (root/'control-manifest.json').exists() and not (root/'control-manifest.json').is_symlink())
        session_path = root/'frame-outbox/stream-session.json'
        if legacy_diagnostic and not session_path.exists() and not session_path.is_symlink():
            outbox = root/'frame-outbox'
            if outbox.is_symlink() or outbox.resolve() != outbox or not outbox.is_dir():
                raise ValueError('Invalid historical diagnostic outbox')
            # Earliest diagnostic spools did not persist this file. The saved
            # closed receipt supplies identity; the remote verifier checks it
            # against every archived frame before returning a result.
            session = {'schema': 'fr5-local-outbox-v1', 'name': root.name,
                       'session': closed.get('session')}
        else:
            session = read('frame-outbox/stream-session.json')
        summary = None if legacy_diagnostic else read('summary.json')
        if (closed.get('schema') != 'fr5-stream-closed-v1'
                or closed.get('local_episode') != str(root)
                or closed.get('error') is not None
                or type(closed.get('accepted_frames')) is not int or closed['accepted_frames'] < 1
                or closed.get('last_durable_sequence') != closed['accepted_frames']-1
                or closed.get('unacknowledged_frames') != 0 or closed.get('unacknowledged_bytes') != 0
                or session.get('schema') != 'fr5-local-outbox-v1' or session.get('name') != root.name
                or session.get('session') != closed.get('session')
                or not stream_protocol.SESSION.fullmatch(str(closed.get('session', '')))
                or (summary is not None and summary.get('scope_sha256') != scope_sha)):
            raise ValueError('Recording is not closed with matching durable acknowledgements')
        spool = SimpleNamespace(session=closed['session'], outbox=root/'frame-outbox',
                                max_bytes=8*1024**2, status=lambda:dict(closed))
        storage = cls(root, scope_sha, spool, required_bytes=MAX_LOCAL_CONTROL)
        if legacy_diagnostic:
            if (not stream_protocol.SHA256.fullmatch(str(closed.get('codec_sha256', '')))
                    or not stream_protocol.SHA256.fullmatch(str(closed.get('protocol_sha256', '')))):
                raise ValueError('Invalid historical codec or protocol hash')
            # No encoding or replay occurs: remote read-only verification
            # checks these hashes against the recording's archived wire files.
            storage.codec_sha256 = closed['codec_sha256']
            storage.protocol_sha256 = closed['protocol_sha256']
            storage._legacy_readonly_recovery = True
        elif (closed.get('codec_sha256') != storage.codec_sha256
              or closed.get('protocol_sha256') != storage.protocol_sha256):
            raise ValueError('Closed recording needs its original compatible codec and protocol')
        return storage

    def _push(self, payloads):
        if not payloads or len(payloads) > MAX_BATCH_FILES or sum(map(len, payloads.values())) > MAX_BATCH:
            raise ValueError("Invalid bounded control batch")
        descriptors = []
        for name, raw in payloads.items():
            p = Path(name)
            if (p.is_absolute() or not p.parts or ".." in p.parts or not isinstance(raw, bytes)
                    or len(raw) > MAX_FILE):
                raise ValueError("Invalid control file")
            descriptors.append({"path": p.as_posix(), "size": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest()})
        request = {"name": self.root.name, "session": self.frame_spool.session,
                   "required_bytes": self.required_bytes, "files": descriptors}
        header = json.dumps(request, separators=(",", ":")).encode()
        payload = struct.pack("!I", len(header)) + header + b"".join(payloads.values())
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
               "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=1",
               REMOTE_HOST, shlex.join([str(REMOTE_BASE.parent / ".venv/bin/python"),
                                       "-c", _REMOTE_PUT])]
        timeout = _control_timeout(len(payload), len(descriptors))
        started = time.monotonic()
        try:
            result = subprocess.run(cmd, input=payload, capture_output=True,
                                    timeout=timeout, check=True)
            receipt = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            detail = {'stage':'control_upload', 'exception_type':type(exc).__name__,
                      'file_count':len(descriptors), 'payload_bytes':len(payload),
                      'timeout_seconds':timeout, 'elapsed_seconds':time.monotonic()-started,
                      'returncode':getattr(exc, 'returncode', None), 'errno':getattr(exc, 'errno', None)}
            stderr = getattr(exc, 'stderr', None)
            if stderr:
                detail['stderr'] = (stderr.decode('utf-8', errors='replace')
                                    if isinstance(stderr, bytes) else str(stderr))[-2048:].strip()
            outbox = self.root/'frame-outbox'
            if outbox.is_dir() and not outbox.is_symlink():
                try:
                    path = outbox/'last-control-error.json'
                    temp = path.with_suffix('.tmp')
                    temp.write_text(json.dumps(detail, indent=2)+'\n')
                    temp.replace(path)
                except OSError:
                    pass  # Preserve the actual transfer exception.
            reason = (f'timeout after {timeout:.1f}s' if isinstance(exc, subprocess.TimeoutExpired)
                      else f'{type(exc).__name__}, exit={detail["returncode"]}')
            if detail.get('stderr'):
                reason += ': ' + detail['stderr']
            raise RuntimeError('4090 did not durably acknowledge the control files: '+reason) from exc
        expected = {item["path"]: item["sha256"] for item in descriptors}
        if receipt.get("session") != self.frame_spool.session or receipt.get("files") != expected:
            raise RuntimeError("Control-file receipt does not match requested bytes")
        return receipt

    def sync_files(self, *paths):
        """Upload scope plus exact named small files over the separate SSH channel."""
        with self._sync_lock:
            payloads = {}
            for value in (self.root / "scope.json", *paths):
                path = Path(value)
                try:
                    relative = path.relative_to(self.root)
                except ValueError as exc:
                    raise ValueError("Cannot sync a file outside the session") from exc
                if (not relative.parts or ".." in relative.parts or path.is_symlink()
                        or path.resolve() != path or not path.is_file()):
                    raise ValueError("Invalid session file")
                payloads[relative.as_posix()] = path.read_bytes()
            if hashlib.sha256(payloads["scope.json"]).hexdigest() != self.scope_sha:
                raise RuntimeError("Scope changed before control-file persistence")
            return self._push(payloads)

    def _control_files(self):
        files = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if relative.parts[0] == "frame-outbox":
                continue
            if path.is_symlink():
                raise ValueError("A control/evidence path is a symlink")
            if not path.is_file():
                continue
            name = relative.as_posix()
            if name in ("frames.stream", "control-manifest.json", "continuous/capture.jsonl",
                        "lerobot/meta/capture.jsonl") or name.startswith("continuous/frames/"):
                continue
            if path.suffix == ".tmp":
                raise RuntimeError("An unfinished control-file write remains")
            if path.stat().st_size > MAX_FILE:
                raise ValueError(f"Control/evidence file exceeds 16 MiB: {name}")
            files.append(path)
        return sorted(files)

    def ensure_local_capacity(self, *, reserve_bytes=0):
        """Bound small control data and checkpoint RGB-D caches during capture."""
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("Invalid local cache reservation")
        total = sum(path.stat().st_size for path in self._control_files())
        if total + reserve_bytes > MAX_LOCAL_CONTROL:
            raise RuntimeError("Local control and checkpoint cache limit would be exceeded")
        free = os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
        if free < reserve_bytes + self.frame_spool.max_bytes + 16 * 1024**2:
            raise RuntimeError("Insufficient local space for bounded outbox and checkpoint")
        return {"control_bytes": total, "control_limit_bytes": MAX_LOCAL_CONTROL,
                "reserved_bytes": reserve_bytes, "outbox_limit_bytes": self.frame_spool.max_bytes}

    def sync_tree(self):
        """Freeze and verify every final control/evidence file in bounded batches."""
        if (self.root / "control-manifest.json").exists():
            raise FileExistsError("A final control snapshot already exists")
        self.ensure_local_capacity()
        files = self._control_files()
        if self.root / "scope.json" not in files or self.root / "summary.json" not in files:
            raise RuntimeError("Final scope and summary are required")
        scope_size = (self.root / "scope.json").stat().st_size
        batch, batch_bytes = [], scope_size
        entries = {}

        def flush():
            nonlocal batch, batch_bytes
            if not batch:
                return
            receipt = self.sync_files(*batch)
            for path in batch:
                name = path.relative_to(self.root).as_posix()
                entries[name] = {"bytes": path.stat().st_size, "sha256": receipt["files"][name]}
            entries["scope.json"] = {"bytes": scope_size, "sha256": receipt["files"]["scope.json"]}
            batch, batch_bytes = [], scope_size

        for path in files:
            if path == self.root / "scope.json":
                continue
            size = path.stat().st_size
            if size + scope_size > MAX_BATCH:
                raise ValueError("A control file and scope exceed the 32 MiB batch limit")
            if batch and (batch_bytes + size > MAX_BATCH or len(batch)+1 >= MAX_BATCH_FILES):
                flush()
            batch.append(path)
            batch_bytes += size
        flush()
        if sorted(files) != self._control_files():
            raise RuntimeError("Control/evidence tree changed during final upload")
        for path in files:
            entry = entries[path.relative_to(self.root).as_posix()]
            if (path.stat().st_size != entry["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]):
                raise RuntimeError("Control/evidence changed after server readback")
        manifest = {"schema": "fr5-final-control-manifest-v1", "session": self.frame_spool.session,
                    "local_episode": str(self.root), "scope_sha256": self.scope_sha,
                    "summary_sha256": entries["summary.json"]["sha256"], "files": entries,
                    "excluded": ["frame-outbox", "frames.stream", "local capture journal placeholders"]}
        path = self.root / "control-manifest.json"
        with path.open("x", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        self.sync_files(path)
        return manifest

    def finalize_control(self):
        """Resume a final manifest upload without recapturing or changing records."""
        path = self.root/'control-manifest.json'
        if not path.exists():
            return self.sync_tree()
        if path.is_symlink():
            raise ValueError('Final control manifest cannot be a link')
        manifest = json.loads(path.read_text())
        if (manifest.get('schema') != 'fr5-final-control-manifest-v1'
                or manifest.get('session') != self.frame_spool.session
                or manifest.get('local_episode') != str(self.root)
                or manifest.get('scope_sha256') != self.scope_sha
                or manifest.get('summary_sha256') != hashlib.sha256((self.root/'summary.json').read_bytes()).hexdigest()
                or not isinstance(manifest.get('files'), dict)):
            raise ValueError('Existing final control manifest does not match this recording')
        files = {p.relative_to(self.root).as_posix():p for p in self._control_files()}
        if set(files) != set(manifest['files']):
            raise ValueError('Closed control files are missing or have changed')
        for name, file in files.items():
            entry = manifest['files'][name]
            if (file.stat().st_size != entry.get('bytes')
                    or hashlib.sha256(file.read_bytes()).hexdigest() != entry.get('sha256')):
                raise ValueError('Closed control file changed: '+name)
        self.sync_files(path)
        return manifest

    def materialize(self, *, timeout=600):
        """Verify and reconstruct a closed stream in a fresh remote directory."""
        from .materialize_stream import (verify_materialized, verify_materialized_legacy_diagnostic,
                                         _finite, _stream_frame_commands, _stream_prediction_sources,
                                         _validate_stream_targets)
        root = str(self.remote_root)
        script = str(self.remote_root / "wire/materialize_stream.py")
        interpreter = "/home/data4/hts/demo1/fr5-learning/.venv/bin/python"
        # Keep the original reconstruction and control checks. Supply missing
        # read-only helper functions when older frozen modules predate them.
        helpers = [(fn.__name__, inspect.getsource(fn)) for fn in
                   (_finite, _stream_frame_commands, _stream_prediction_sources, _validate_stream_targets,
                    verify_materialized_legacy_diagnostic)]
        runner = ("import importlib.util,json,math,sys\nfrom pathlib import Path\n"
                  "sys.dont_write_bytecode=True\n"
                  "spec=importlib.util.spec_from_file_location('saved_materializer',sys.argv[1])\n"
                  "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n"
                  "module.__dict__.setdefault('math',math)\n"
                  "for name,source in "+repr(helpers)+":\n"
                  " if name not in module.__dict__:exec(source,module.__dict__)\n"
                  "exec("+repr(inspect.getsource(verify_materialized))+",module.__dict__)\n"
                  "root=Path(sys.argv[2])\n"
                  "if "+repr(bool(getattr(self, '_legacy_readonly_recovery', False)))+" and not (root/'materialized').exists():\n"
                  " raise ValueError('Historical diagnostic recovery only verifies an existing materialization')\n"
                  "if (root/'materialized').exists():\n"
                  " scope=json.loads((root/'scope.json').read_text())\n"
                  " if not (root/'control-manifest.json').exists() and scope.get('diagnostic_only') is True:\n"
                  "  result=module.verify_materialized_legacy_diagnostic(root)\n"
                  " else:result=module.verify_materialized(root)\n"
                  "else:result=module.materialize(root)\n"
                  "print(json.dumps(result,ensure_ascii=False),flush=True)\n")
        command = " ".join(map(shlex.quote, (interpreter, '-c', runner, script, root)))
        try:
            result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                                     REMOTE_HOST, command], capture_output=True, timeout=timeout, check=True)
            receipt = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise RuntimeError("Remote stream materialization did not complete") from exc
        if (receipt.get("session") != self.frame_spool.session
                or receipt.get("codec_sha256") != self.codec_sha256
                or receipt.get("protocol_sha256") != self.protocol_sha256
                or receipt.get("frames") != self.frame_spool.status()["accepted_frames"]):
            raise RuntimeError("Remote materialization receipt mismatch")
        if (self.root / "control-manifest.json").exists():
            # A stream evaluation may explicitly contain undelivered callback
            # records. Preserve its replay and mark the alignment incomplete;
            # never apply this exception to demonstration materialization.
            scope = json.loads((self.root / 'scope.json').read_text())
            summary = json.loads((self.root / 'summary.json').read_text())
            policy_contract = scope.get('policy_contract')
            incomplete_evaluation = (scope.get('dataset_role') == 'policy_evaluation'
                and isinstance(policy_contract, dict)
                and policy_contract.get('name') == 'fr5-joint-position-stream-v1'
                and summary.get('wire_callbacks_drained') is False
                and receipt.get('dataset_role') == 'policy_evaluation'
                and receipt.get('task_success_verified') is False)
            readonly_evaluation = (scope.get('dataset_role') == 'policy_evaluation'
                and isinstance(policy_contract, dict)
                and policy_contract.get('name') == 'fr5-joint-position-stream-v1'
                and summary.get('read_only_diagnostic') is True
                and summary.get('model_execution') is False
                and summary.get('wire_callbacks_drained') is None
                and receipt.get('dataset_role') == 'policy_evaluation'
                and receipt.get('task_success_verified') is False)
            expected_alignment = ('not_applicable_read_only' if readonly_evaluation
                                  else 'incomplete_missing_wire_callbacks' if incomplete_evaluation
                                  else 'verified_against_wire_and_frame_clocks')
            if (receipt.get("control_verification") != "verified"
                    or receipt.get("target_alignment") != expected_alignment
                    or receipt.get("scope_sha256") != hashlib.sha256((self.root / "scope.json").read_bytes()).hexdigest()
                    or receipt.get("summary_sha256") != hashlib.sha256((self.root / "summary.json").read_bytes()).hexdigest()):
                raise RuntimeError("Final local and materialized scope/summary, evidence or targets disagree")
        return receipt

    def release_checkpoint_caches(self, receipt):
        """Release only verified view-*/scene.npz caches, retaining review files."""
        manifest_path = self.root / "control-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (receipt.get("control_verification") != "verified"
                or receipt.get("target_alignment") != "verified_against_wire_and_frame_clocks"
                or receipt.get("session") != self.frame_spool.session
                or receipt.get("scope_sha256") != manifest.get("scope_sha256")
                or receipt.get("summary_sha256") != manifest.get("summary_sha256")):
            raise RuntimeError("Cannot release caches without a verified final materialization")
        protected = set()
        for document in (json.loads((self.root / "scope.json").read_text()),
                         json.loads((self.root / "summary.json").read_text())):
            for value in document.get("evidence", []):
                if not isinstance(value, str):
                    continue
                path = Path(value)
                if path.is_absolute():
                    try:
                        path = path.relative_to(self.root)
                    except ValueError:
                        continue
                protected.add(path.as_posix())
        caches = []
        for name, entry in manifest["files"].items():
            relative = Path(name)
            if (len(relative.parts) != 2 or not relative.parts[0].startswith("view-")
                    or relative.name != "scene.npz" or name in protected):
                continue
            path = self.root / relative
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]):
                raise RuntimeError("Checkpoint cache changed since remote verification")
            caches.append({"path": name, "bytes": entry["bytes"], "sha256": entry["sha256"]})
        if not caches:
            return []
        ledger = self.frame_spool.outbox / "released-checkpoint-caches.json"
        with ledger.open("x", encoding="utf-8") as file:
            json.dump({"schema": "fr5-verified-cache-release-v1", "session": self.frame_spool.session,
                       "remote_materialized": str(self.remote_root / "materialized"),
                       "control_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "caches": caches}, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        from .stream_protocol import _sync_dir
        _sync_dir(ledger.parent)
        for item in caches:
            path = self.root / item["path"]
            path.unlink()
            _sync_dir(path.parent)
        return caches

