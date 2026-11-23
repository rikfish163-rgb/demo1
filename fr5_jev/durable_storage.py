"""Fail-closed storage for new teacher recordings on 4090's data4 filesystem.

The existing recorder writes directly through the system GVFS SFTP mount.
Nothing in this module connects to a robot or camera.  Every session must use
a new directory inside ``recording-spool``; accepted datasets are not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid


REMOTE_HOST = "4090"
REMOTE_BASE = PurePosixPath("/home/data4/hts/demo1/fr5-learning/recording-spool")
GVFS_ROOT = Path("/run/user") / str(os.getuid()) / "gvfs"
MOUNT_ROOT = GVFS_ROOT / "sftp:host=4090"
LOCAL_BASE = MOUNT_ROOT.joinpath(*REMOTE_BASE.parts[1:])
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
FPS = 6
PROBE_FRAMES = 3
PROBE_LIMIT_S = 0.9 / FPS
PROBE_TIMEOUT_S = 20.0


# This helper only reads and fsyncs explicitly named regular files. It never
# creates, renames, removes or changes an accepted recording on the server.
_REMOTE_CHECK = r'''
import hashlib,json,os,stat,sys
from pathlib import Path
q=json.load(sys.stdin)
base=Path('/home/data4/hts/demo1/fr5-learning/recording-spool')
root=Path(q['root'])
if os.path.realpath(base)!=str(base) or os.path.realpath(root)!=str(root):
 raise ValueError('spool directory must not contain symlinks')
if root.parent!=base or not root.is_dir():
 raise ValueError('not a direct recording-spool child')
free=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
if free<q.get('required_bytes',0):
 raise OSError('insufficient remote free space')
for name,expected in q['files'].items():
 p=Path(name)
 if p.is_absolute() or not p.parts or '..' in p.parts:
  raise ValueError('invalid relative file')
 target=root/p
 if not os.path.realpath(target).startswith(str(root)+'/'):
  raise ValueError('file escaped recording root')
 fd=os.open(target,os.O_RDONLY | getattr(os,'O_NOFOLLOW',0))
 try:
  if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('not a regular file')
  os.fsync(fd)
  with os.fdopen(fd,'rb',closefd=False) as f:
   digest=hashlib.sha256()
   for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
  if digest.hexdigest()!=expected:raise ValueError('remote file hash mismatch: '+name)
 finally:os.close(fd)
dirfd=os.open(root,os.O_RDONLY | getattr(os,'O_DIRECTORY',0))
try:os.fsync(dirfd)
finally:os.close(dirfd)
print(json.dumps({'free_bytes':free,'files_verified':len(q['files'])}))
'''


def _mounted():
    """Require the system GVFS FUSE mount, not an ordinary local directory."""
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            if left.split()[4] == str(GVFS_ROOT) and right.split()[0] == "fuse.gvfsd-fuse":
                return MOUNT_ROOT.is_dir()
    except (OSError, ValueError, IndexError):
        return False
    return False


def ensure_mount():
    """Mount the existing SFTP backend and create only the new spool parent."""
    if not _mounted():
        # The Conda GLib environment can hide the installed system GVFS module.
        env = os.environ.copy()
        for key in ("LD_LIBRARY_PATH", "GIO_EXTRA_MODULES", "GIO_MODULE_DIR"):
            env.pop(key, None)
        try:
            subprocess.run(["/usr/bin/gio", "mount", "sftp://4090/"], env=env,
                           check=True, capture_output=True, timeout=12)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Cannot mount the 4090 SFTP recording store") from exc
    if not _mounted():
        raise RuntimeError("The 4090 GVFS SFTP mount is not available")
    LOCAL_BASE.mkdir(exist_ok=True)
    if not LOCAL_BASE.is_dir() or LOCAL_BASE.is_symlink():
        raise RuntimeError("Recording spool parent is unavailable or a symlink")
    return LOCAL_BASE


def new_recording_path(name):
    """Return a fresh recording path; the existing preparer creates it."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError("Recording name must be a simple filename")
    base = ensure_mount()
    path = base / name
    if path.exists() or path.is_symlink():
        raise FileExistsError("Choose a new recording name")
    return path


def _server_check(remote_root, files, *, required_bytes=0):
    request = {"root": str(remote_root), "files": files, "required_bytes": required_bytes}
    try:
        response = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
             "-o", "ServerAliveInterval=3", "-o", "ServerAliveCountMax=1",
             REMOTE_HOST, "python3 -c " + shlex.quote(_REMOTE_CHECK)],
            input=json.dumps(request), text=True, capture_output=True, timeout=8, check=True)
        result = json.loads(response.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("4090 could not verify and sync recording storage") from exc
    if result.get("files_verified") != len(files) or result.get("free_bytes", -1) < required_bytes:
        raise RuntimeError("Incomplete remote storage verification")
    return result


def _fsync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _probe_frame(directory, index, color, depth, logs):
    """Exercise the same two files and two flushed journals as one frame."""
    start = time.monotonic()
    for extension, payload in (("png", color), ("npz", depth)):
        path = directory / f"{index:06d}.{extension}"
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    for log in logs:
        log.write((json.dumps({"synthetic_probe_frame": index}) + "\n").encode())
        log.flush()
        os.fsync(log.fileno())
    return time.monotonic() - start


def _bench(base, remote_root, scope_sha, required_bytes):
    # Probe data is synthetic and outside the episode; it is never a recording.
    name = ".durability-probe-" + uuid.uuid4().hex
    directory = base / name
    directory.mkdir()
    color, depth = os.urandom(300000), os.urandom(130000)
    durations = []
    try:
        with (directory / "capture.jsonl").open("xb") as capture, \
                (directory / "teacher.jsonl").open("xb") as teacher:
            for index in range(PROBE_FRAMES):
                durations.append(_probe_frame(directory, index, color, depth, (capture, teacher)))
                if durations[-1] > PROBE_LIMIT_S:
                    break
        # Independently verify the mount against SSH and flush the server file.
        # The probe is deliberately a sibling, so check it using its own root.
        _server_check(REMOTE_BASE / name, {"000000.png": hashlib.sha256(color).hexdigest()},
                      required_bytes=required_bytes)
        _server_check(remote_root, {"scope.json": scope_sha}, required_bytes=required_bytes)
        if len(durations) != PROBE_FRAMES or max(durations) > PROBE_LIMIT_S:
            raise RuntimeError(
                f"Remote recording is too slow for {FPS} fps: "
                f"{max(durations):.3f}s per synthetic frame; limit {PROBE_LIMIT_S:.3f}s")
        return durations
    finally:
        # Only our exclusively-created, synthetic probe is removed.
        shutil.rmtree(directory)


def _bounded(call, timeout=PROBE_TIMEOUT_S):
    """Do not let an unresponsive mounted filesystem hang startup forever."""
    result, finished = {}, threading.Event()

    def run():
        try:
            result["value"] = call()
        except BaseException as exc:
            result["error"] = exc
        finally:
            finished.set()

    threading.Thread(target=run, name="fr5-storage-preflight", daemon=True).start()
    if not finished.wait(timeout):
        raise TimeoutError("Recording storage preflight timed out")
    if "error" in result:
        raise result["error"]
    return result["value"]


class DurableStorage:
    def __init__(self, root, scope_sha):
        self.root = Path(root)
        self.scope_sha = scope_sha
        self.remote_root = REMOTE_BASE / self.root.name

    @classmethod
    def preflight(cls, root, scope_sha, *, required_bytes):
        if not re.fullmatch(r"[0-9a-f]{64}", scope_sha):
            raise ValueError("Invalid scope hash")
        if isinstance(required_bytes, bool) or not isinstance(required_bytes, int) or required_bytes <= 0:
            raise ValueError("A positive capture storage budget is required")

        def check():
            if not _mounted():
                raise RuntimeError("The 4090 SFTP mount is not available")
            root_path = Path(root)
            if (not NAME.fullmatch(root_path.name) or root_path.parent != LOCAL_BASE
                    or root_path.is_symlink() or not root_path.is_dir()
                    or root_path.resolve() != root_path):
                raise RuntimeError("Recording root must be a direct, non-symlink child of the 4090 spool")
            if any((root_path / name).exists() for name in
                   ("SESSION_STARTED", "summary.json", "continuous", "lerobot", "storage-preflight.json")):
                raise RuntimeError("Recording root is already claimed or contains recording data")
            scope_file = root_path / "scope.json"
            raw = scope_file.read_bytes()
            if hashlib.sha256(raw).hexdigest() != scope_sha:
                raise RuntimeError("Scope changed before storage preflight")
            scope = json.loads(raw)
            for key in ("geometry_path", "manual_action_plan_path", "lateral_geometry_path"):
                if key in scope:
                    p = Path(scope[key])
                    if not p.is_absolute() or not p.resolve().is_relative_to(root_path) or not p.is_file():
                        raise RuntimeError(f"{key} must be stored inside the new recording root")
            for item in scope.get("evidence", []):
                if isinstance(item, str) and Path(item).is_absolute():
                    p = Path(item)
                    if not p.resolve().is_relative_to(root_path) or not p.is_file():
                        raise RuntimeError("Absolute scope evidence must be inside the recording root")
            storage = cls(root_path, scope_sha)
            durations = _bench(LOCAL_BASE, storage.remote_root, scope_sha, required_bytes)
            receipt = {"schema": "fr5-durable-recording-preflight-v1", "host": REMOTE_HOST,
                       "server_root": str(storage.remote_root), "scope_sha256": scope_sha,
                       "required_bytes": required_bytes, "probe_frame_seconds": durations,
                       "probe_frame_limit_s": PROBE_LIMIT_S, "checked_unix": time.time()}
            receipt_path = root_path / "storage-preflight.json"
            with receipt_path.open("x", encoding="utf-8") as file:
                json.dump(receipt, file, indent=2, allow_nan=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            storage.sync_files(receipt_path)
            return storage, durations

        storage, durations = _bounded(check)
        storage.probe_frame_seconds = durations
        return storage

    def sync_files(self, *paths):
        """Verify exact bytes via SSH and fsync them on the server, with a timeout."""
        files = {"scope.json": self.scope_sha}
        for value in paths:
            path = Path(value)
            try:
                relative = path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError("Cannot sync a file outside this recording") from exc
            if not relative.parts or ".." in relative.parts or path.is_symlink():
                raise ValueError("Invalid recording file")
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return _server_check(self.remote_root, files)


def probe_server():
    """Test a synthetic 1800-second storage budget without opening devices."""
    name = "synthetic-preflight-" + uuid.uuid4().hex
    root = new_recording_path(name)
    root.mkdir()
    scope = root / "scope.json"
    timed_out = False
    try:
        scope.write_text(json.dumps({"evidence": ["synthetic storage probe only"]}))
        digest = hashlib.sha256(scope.read_bytes()).hexdigest()
        required = 512 * 1024**2 + 1800 * FPS * 512 * 1024
        result = DurableStorage.preflight(root, digest, required_bytes=required)
        return {"status": "passed", "frame_seconds": result.probe_frame_seconds,
                "frame_limit_s": PROBE_LIMIT_S, "required_bytes": required}
    except TimeoutError:
        # The timed-out daemon can still be touching its own probe files.
        timed_out = True
        raise
    finally:
        if not timed_out:
            for name in ("storage-preflight.json", "scope.json"):
                try:
                    (root / name).unlink()
                except FileNotFoundError:
                    pass
            root.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="A new recording directory name")
    parser.add_argument("--probe", action="store_true", help="Test storage without a camera or robot")
    parser.add_argument("--stream", action="store_true", help="Plan a fresh local outbox for the diagnostic stream")
    args = parser.parse_args()
    if args.probe == (args.name is not None):
        parser.error("Specify either a new recording name or --probe")
    if args.probe and args.stream:
        parser.error("The stream has a separate read-only diagnostic")
    try:
        if args.probe:
            result = _bounded(probe_server, timeout=PROBE_TIMEOUT_S + 15)
        else:
            if args.stream:
                from .stream_storage import new_stream_recording_path
                path = new_stream_recording_path(args.name)
            else:
                path = _bounded(lambda: new_recording_path(args.name))
            result = {"local_recording_path": str(path),
                      "server_recording_path": str(REMOTE_BASE / args.name),
                      "note": ("Fresh stream path; execution requires current verified readiness and a fresh server preflight."
                               if args.stream else "Prepare a fresh scope here; execution repeats the storage preflight.")}
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
