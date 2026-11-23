"""本机采集网页的数据、完整回放与远程媒体读取。

所有采集与回放文件留在 data4；episode 参数是 recording-spool 的直接子目录名。
本模块不控制设备，也不写入用户审核或采集记录。
"""
from __future__ import annotations

import shlex
import subprocess

from fr5_runtime.system import REMOTE, remote_json, simple_name, ssh_args


# 这些代码由现有 remote_json 在服务器的 data4 虚拟环境中执行。
# 路径只由固定根目录和经过 simple_name 校验的 episode 名组成。
_REMOTE_COMMON = r'''
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


def directory(path):
    return path.is_dir() and not path.is_symlink() and path.resolve() == path


def regular_file(path):
    return (directory(path.parent) and path.is_file() and not path.is_symlink()
            and path.resolve() == path)


def read_object(path):
    if not regular_file(path):
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def episode_dir(root, name):
    spool = root / 'recording-spool'
    path = spool / name
    if not directory(spool) or not directory(path):
        raise ValueError('采集目录不存在')
    return path


def materialized_dir(root, name):
    path = episode_dir(root, name) / 'materialized'
    if not directory(path):
        raise ValueError('采集数据尚未物化')
    scope = read_object(path / 'scope.json')
    if scope.get('whole_task') is not True and scope.get('dataset_role') != 'policy_evaluation':
        raise ValueError('不是完整任务或策略实机实验记录')
    return path


def capture_rows(source):
    path = source / 'continuous' / 'capture.jsonl'
    if not regular_file(path):
        raise ValueError('缺少采集帧记录')
    materialization = read_object(source / 'stream-materialization.json')
    if materialization.get('schema') != 'fr5-stream-materialization-v1':
        raise ValueError('缺少物化帧校验信息')
    times, images = [], []
    with path.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if not isinstance(row, dict) or row.get('frame_index') != index:
                raise ValueError('采集帧序号无效')
            camera = row.get('camera')
            timestamp = camera.get('source_timestamp_monotonic') if isinstance(camera, dict) else None
            if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                    or (times and timestamp <= times[-1])):
                raise ValueError('采集帧时间无效')
            image_name = row.get('rgb_file')
            if image_name != 'frames/%06d.jpg' % index:
                raise ValueError('采集帧路径无效')
            times.append(float(timestamp))
            codec = row.get('media_codec')
            digest = codec.get('rgb_sha256') if isinstance(codec, dict) else None
            if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('采集图片缺少校验信息')
            images.append((image_name, digest))
    if not times:
        raise ValueError('没有采集帧')
    if (type(materialization.get('frames')) is not int or materialization['frames'] != len(times)
            or materialization.get('capture_journal_sha256') != sha256(path)):
        raise ValueError('采集帧清单不完整或已改变')
    return times, images


def decoded_video(video, times):
    import av
    count = 0
    maximum_error = 0.0
    with av.open(str(video), mode='r') as container:
        if len(container.streams.video) != 1:
            raise ValueError('回放视频轨道无效')
        stream = container.streams.video[0]
        if (stream.codec_context.name != 'h264' or stream.width != 640 or stream.height != 360):
            raise ValueError('回放视频格式无效')
        for frame in container.decode(stream):
            if count >= len(times) or frame.pts is None or frame.time_base is None:
                raise ValueError('回放帧数或时间无效')
            error = abs(float(frame.pts * frame.time_base) - (times[count] - times[0]))
            if not math.isfinite(error) or error > .002:
                raise ValueError('回放时间与采集时间不一致')
            maximum_error = max(maximum_error, error)
            count += 1
    if count != len(times):
        raise ValueError('回放帧数不完整')
    return maximum_error


def valid_video(root, name, source, times=None):
    reviews = root / 'reviews'
    review_dir = reviews / name
    video, manifest = review_dir / 'full-process.mp4', review_dir / 'review-video.json'
    if not directory(reviews) or not directory(review_dir) or not regular_file(video):
        return None
    proof = read_object(manifest)
    try:
        if times is None:
            times, _ = capture_rows(source)
        if (proof.get('source') != str(source) or proof.get('video') != str(video)
                or proof.get('frames') != len(times) or proof.get('video_bytes') != video.stat().st_size
                or proof.get('all_frames_preserved') is not True or proof.get('waits_trimmed') is not False
                or not isinstance(proof.get('video_sha256'), str)
                or proof['video_sha256'] != sha256(video)):
            return None
        decoded_video(video, times)
        return proof
    except Exception:
        return None


def finite_number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def iso_time(value):
    value = finite_number(value)
    if value is None:
        return None
    try:
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
    except (OverflowError, OSError, ValueError):
        return None
'''

_INVENTORY_CODE = _REMOTE_COMMON + r'''
q = json.load(sys.stdin)
root = Path(q['root'])
result = {'storage': {'ok': False, 'free_bytes': None},
          'counts': {'success': 0, 'pending': 0, 'total': 0}, 'episodes': []}
try:
    if not directory(root):
        raise OSError('存储目录不可用')
    stats = os.statvfs(root)
    spool = root / 'recording-spool'
    if not directory(spool):
        raise OSError('采集目录不可用')
    result['storage'] = {'ok': True, 'free_bytes': stats.f_bavail * stats.f_frsize}
    for path in sorted(spool.iterdir()):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', path.name) or not directory(path):
            continue
        materialized = path / 'materialized'
        has_materialized = directory(materialized)
        source = materialized if has_materialized else path
        scope = read_object(source / 'scope.json')
        if scope.get('whole_task') is not True:
            # A still-in-progress or interrupted materialization can lack its scope.
            if has_materialized:
                scope = read_object(path / 'scope.json')
                if scope.get('whole_task') is True:
                    source, has_materialized = path, False
            if scope.get('whole_task') is not True:
                continue
        summary = read_object(source / 'summary.json')
        proof = read_object(source / 'stream-materialization.json') if has_materialized else {}
        scope_path, summary_path = source / 'scope.json', source / 'summary.json'
        if has_materialized and not regular_file(summary_path):
            has_materialized = False
        scope_hash = sha256(scope_path) if regular_file(scope_path) else None
        summary_hash = sha256(summary_path) if regular_file(summary_path) else None
        materialization_valid = (has_materialized and proof.get('schema') == 'fr5-stream-materialization-v1'
                                 and proof.get('scope_sha256') == scope_hash
                                 and proof.get('summary_sha256') == summary_hash)
        review = read_object(source / 'user-review.json') if has_materialized else {}
        review_valid = (materialization_valid and review.get('schema') in
                        ('fr5-user-review-v1', 'fr5-policy-user-review-v1')
                        and review.get('reviewer') == 'user' and review.get('source') == 'direct_user_message'
                        and review.get('verdict') in ('success', 'failure')
                        and isinstance(review.get('user_statement'), str) and bool(review['user_statement'].strip())
                        and review.get('scope_sha256') == scope_hash
                        and review.get('summary_sha256') == summary_hash)
        outcome = summary.get('outcome')
        receipt = read_object(path / 'stream-closed.json')
        if not has_materialized:
            status = ('saving' if type(receipt.get('accepted_frames')) is int
                      and receipt['accepted_frames'] > 0 else 'interrupted')
        elif review_valid and review['verdict'] == 'success' and outcome == 'pending_review' and not summary.get('failed'):
            status = 'success'
        elif review_valid and review['verdict'] == 'failure':
            status = 'failure'
        elif materialization_valid and outcome == 'pending_review' and not summary.get('failed'):
            status = 'pending_review'
        elif outcome in ('failure', 'failed', 'unsuccessful', 'unsuccessful_observed'):
            status = 'failure'
        else:
            status = 'interrupted'
        capture = summary.get('continuous_capture')
        capture = capture if isinstance(capture, dict) else {}
        frames = proof.get('frames') if has_materialized else receipt.get('accepted_frames')
        if type(frames) is not int or frames < 0:
            frames = capture.get('frames')
        if type(frames) is not int or frames < 0:
            frames = None
        times = None
        duration = None
        if has_materialized:
            try:
                times, _ = capture_rows(source)
                frames = len(times)
                duration = times[-1] - times[0]
            except (OSError, ValueError, TypeError, KeyError, UnicodeError):
                pass
        video = None
        if has_materialized and times is not None and valid_video(root, path.name, source, times):
            video = '/media/' + path.name + '.mp4'
        spec = scope.get('task_spec')
        spec = spec if isinstance(spec, dict) else {}
        hold = finite_number(scope.get('hold_seconds'))
        if hold is None:
            hold = finite_number(spec.get('hold_seconds'))
        recorded_holds = []
        for entry in summary.get('hold_records', []):
            hold_path = entry.get('path') if isinstance(entry, dict) else None
            if isinstance(hold_path, str) and re.fullmatch(r'hold-\d+\.json', hold_path):
                row = read_object(source / hold_path)
                value = finite_number(row.get('actual_hover_from_lift_completion_s', row.get('actual_duration_s')))
                if value is not None:
                    recorded_holds.append(value)
        actual_hold = sum(recorded_holds) if recorded_holds else None
        retreat = scope.get('retreat_mode', summary.get('retreat_mode'))
        result['episodes'].append({
            'id': path.name, 'created_at': iso_time(scope.get('prepared_unix')),
            'status': status, 'frames': frames, 'duration_s': duration,
            'hold_seconds': hold, 'actual_hold_seconds': actual_hold,
            'retreat_mode': retreat if isinstance(retreat, str) else None,
            'video_url': video, 'reviewed': bool(review_valid),
            'can_review':bool(materialization_valid),
            'review_statement': review['user_statement'] if review_valid else None})
    result['episodes'].sort(key=lambda item: (item['created_at'] or '', item['id']), reverse=True)
    result['counts'] = {'success': sum(item['status'] == 'success' for item in result['episodes']),
                        'pending': sum(item['status'] == 'pending_review' for item in result['episodes']),
                        'total': len(result['episodes'])}
except OSError:
    result = {'storage': {'ok': False, 'free_bytes': None},
              'counts': {'success': 0, 'pending': 0, 'total': 0}, 'episodes': []}
print(json.dumps(result, ensure_ascii=False, allow_nan=False))
'''

_RENDER_CODE = _REMOTE_COMMON + r'''
import fcntl
from fractions import Fraction
import io
import tempfile

q = json.load(sys.stdin)
root, name = Path(q['root']), q['episode']
source = materialized_dir(root, name)
times, images = capture_rows(source)
reviews = root / 'reviews'
review_dir = reviews / name
video, manifest = review_dir / 'full-process.mp4', review_dir / 'review-video.json'
existing = valid_video(root, name, source, times)
if existing:
    print(json.dumps(existing, ensure_ascii=False))
    sys.exit(0)
if not directory(root):
    raise ValueError('存储目录不可用')
reviews.mkdir(mode=0o700, exist_ok=True)
if not directory(reviews):
    raise ValueError('回放目录无效')
review_dir.mkdir(mode=0o700, exist_ok=True)
if not directory(review_dir):
    raise ValueError('回放目录无效')
lock_fd = os.open(str(review_dir / '.render.lock'), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
try:
    with os.fdopen(lock_fd, 'a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = valid_video(root, name, source, times)
        if existing:
            print(json.dumps(existing, ensure_ascii=False))
            sys.exit(0)
        if video.exists() or video.is_symlink() or manifest.exists() or manifest.is_symlink():
            raise ValueError('已有回放文件未通过校验，未覆盖')
        import av
        from PIL import Image
        ticks = [round((timestamp - times[0]) * 1000) for timestamp in times]
        if any(right <= left for left, right in zip(ticks, ticks[1:])):
            raise ValueError('帧时间不足一毫秒，无法无损表示')
        temp_video = temp_manifest = None
        try:
            fd, temp_video = tempfile.mkstemp(prefix='.full-process-', suffix='.mp4', dir=review_dir)
            os.close(fd)
            with av.open(temp_video, mode='w', format='mp4', options={'movflags': '+faststart'}) as container:
                stream = container.add_stream('libx264', rate=6)
                stream.width, stream.height, stream.pix_fmt = 640, 360, 'yuv420p'
                stream.time_base = Fraction(1, 1000)
                stream.codec_context.time_base = Fraction(1, 1000)
                stream.options = {'preset': 'veryfast', 'crf': '18', 'bf': '0'}
                for tick, (image_name, expected_hash) in zip(ticks, images):
                    image_path = source / 'continuous' / image_name
                    if not regular_file(image_path):
                        raise ValueError('采集图片不存在')
                    image_bytes = image_path.read_bytes()
                    if hashlib.sha256(image_bytes).hexdigest() != expected_hash:
                        raise ValueError('采集图片校验失败')
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        if image.size != (640, 360) or image.mode != 'RGB':
                            raise ValueError('采集图片不是 RGB 640x360')
                        frame = av.VideoFrame.from_image(image)
                    frame.pts, frame.time_base = tick, Fraction(1, 1000)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with open(temp_video, 'rb') as file:
                os.fsync(file.fileno())
            max_error = decoded_video(Path(temp_video), times)
            if capture_rows(source) != (times, images):
                raise ValueError('回放生成时采集帧清单发生改变')
            with av.open(temp_video, mode='r') as container:
                video_duration = (container.duration / av.time_base if container.duration is not None else None)
            proof = {'source': str(source), 'video': str(video), 'frames': len(times),
                     'source_duration_s': times[-1] - times[0], 'video_duration_s': video_duration,
                     'video_bytes': os.stat(temp_video).st_size, 'all_frames_preserved': True,
                     'waits_trimmed': False, 'max_timestamp_error_s': max_error,
                     'video_sha256': sha256(Path(temp_video))}
            fd, temp_manifest = tempfile.mkstemp(prefix='.review-video-', suffix='.json', dir=review_dir)
            with os.fdopen(fd, 'w', encoding='utf-8') as file:
                json.dump(proof, file, ensure_ascii=False, indent=2, allow_nan=False)
                file.write('\n')
                file.flush()
                os.fsync(file.fileno())
            # Link installation is atomic and cannot replace an existing file.
            os.link(temp_video, video, follow_symlinks=False)
            os.link(temp_manifest, manifest, follow_symlinks=False)
            dir_fd = os.open(str(review_dir), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            print(json.dumps(proof, ensure_ascii=False, allow_nan=False))
        finally:
            for temporary in (temp_video, temp_manifest):
                if temporary is not None:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
except BaseException:
    raise
'''

_MEDIA_METADATA_CODE = _REMOTE_COMMON + r'''
q = json.load(sys.stdin)
root, name = Path(q['root']), q['episode']
reviews, review_dir = root / 'reviews', root / 'reviews' / name
video = review_dir / 'full-process.mp4'
if not directory(root) or not directory(reviews) or not directory(review_dir) or not regular_file(video):
    print(json.dumps({'error': '回放文件不存在'}))
else:
    info = video.stat()
    print(json.dumps({'size': info.st_size, 'mtime': info.st_mtime}))
'''

_MEDIA_STREAM_CODE = r'''
import os
from pathlib import Path
import re
import stat
import sys

root, name, start, end = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
if (root.resolve() != root or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name)
        or start < 0 or end < start):
    raise ValueError('invalid request')
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
reviews_fd = os.open(str(root / 'reviews'), flags)
try:
    episode_fd = os.open(name, flags, dir_fd=reviews_fd)
    try:
        fd = os.open('full-process.mp4', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=episode_fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or end >= info.st_size:
                raise ValueError('invalid range')
            os.lseek(fd, start, os.SEEK_SET)
            remaining = end - start + 1
            with os.fdopen(fd, 'rb', closefd=False) as file:
                while remaining:
                    chunk = file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError('unexpected end of file')
                    sys.stdout.buffer.write(chunk)
                    remaining -= len(chunk)
                sys.stdout.buffer.flush()
        finally:
            os.close(fd)
    finally:
        os.close(episode_fd)
finally:
    os.close(reviews_fd)
'''


def _episode(value):
    try:
        return simple_name(value)
    except ValueError:
        raise ValueError('无效的采集记录名称') from None


def inventory():
    """只统计 recording-spool 中 whole_task 的采集记录。"""
    try:
        return remote_json(_INVENTORY_CODE, {'root': REMOTE}, timeout=180)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError):
        return {'storage': {'ok': False, 'free_bytes': None},
                'counts': {'success': 0, 'pending': 0, 'total': 0}, 'episodes': []}


def render_replay(episode):
    """在 data4 生成并校验完整回放；已有文件不会被覆盖。"""
    name = _episode(episode)
    try:
        proof = remote_json(_RENDER_CODE, {'root': REMOTE, 'episode': name}, timeout=3600)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError):
        raise RuntimeError('回放生成或校验失败；已有文件未覆盖') from None
    return {**proof, 'video_url': '/media/' + name + '.mp4'}


def media_metadata(episode):
    """返回服务器固定回放路径的大小和修改时间。"""
    name = _episode(episode)
    try:
        result = remote_json(_MEDIA_METADATA_CODE, {'root': REMOTE, 'episode': name}, timeout=30)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError):
        raise RuntimeError('无法读取回放文件信息') from None
    if result.get('error'):
        raise FileNotFoundError(result['error'])
    return result


def open_media(episode, start, end):
    """返回 SSH 子进程，stdout 流式提供 [start, end] 的视频字节。"""
    name = _episode(episode)
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ValueError('无效的视频字节范围')
    command = shlex.join([REMOTE + '/.venv/bin/python', '-c', _MEDIA_STREAM_CODE,
                          REMOTE, name, str(start), str(end)])
    try:
        return subprocess.Popen(ssh_args(command), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        raise RuntimeError('无法打开回放视频') from None
