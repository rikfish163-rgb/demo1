"""Local read-only Gemini 335 / FR5 monitor. No robot motion commands."""
import io, json, threading, time, xmlrpc.client
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np
import pyorbbecsdk
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, AlignFilter, OBStreamType, OBFrameAggregateOutputMode
from fr5_jev.rgbd_geometry import color_in_depth_grid
from fr5_jev.calibration import load_readonly_calibration


def calibration_status():
    try:
        calibration = load_readonly_calibration(camera_serial='CP1E54200015')
    except (OSError, ValueError, KeyError) as exc:
        return f'手眼标定未加载：{type(exc).__name__}'
    return ('已加载 J1 手眼标定，样本区域验证通过；'
            '相机和底座需保持不变，夹持中心另行确认')


state = {'camera': {'ok': False}, 'robot': {'ok': False}, 'calibration': calibration_status()}
images = {}
snapshot = {}
lock = threading.Lock()
PREVIEW_INTERVAL_S = 0.2
FAST_WIDTH, FAST_HEIGHT = 640, 360
live_dir = Path.home() / '.local/state/fr5-system/monitor'
live_dir.mkdir(parents=True, exist_ok=True)
class Transport(xmlrpc.client.Transport):
    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = 3
        return conn

def robot_loop():
    r = xmlrpc.client.ServerProxy('http://192.168.58.2:20003', transport=Transport())
    while True:
        try:
            j = r.GetActualJointPosDegree(1)
            p = r.GetActualTCPPose(1)
            if j[0] or p[0]: raise RuntimeError(f'返回码 {j[0]}, {p[0]}')
            value = dict(ok=True, received_at=time.time(), joints_deg=j[1:], tcp_mm_deg=p[1:])
        except Exception as e: value = dict(ok=False, error=str(e), received_at=time.time())
        with lock: state['robot'] = value
        time.sleep(.5)

def profile_parameters(profile):
    """Read actual frame profile parameters; no device writes."""
    profile = profile.as_video_stream_profile()
    intrinsic, distortion = profile.get_intrinsic(), profile.get_distortion()
    return {'intrinsic': {k: getattr(intrinsic, k) for k in ['fx', 'fy', 'cx', 'cy', 'width', 'height']},
            'distortion': {k: getattr(distortion, k) for k in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']},
            'distortion_model': str(distortion.model), 'fps': profile.get_fps(),
            'format': str(profile.get_format())}


def frame_timing(frame):
    return {'index': frame.get_index(), 'device_timestamp_us': frame.get_timestamp_us(),
            'system_timestamp_us': frame.get_system_timestamp_us()}


def snapshot_packet(color, depth, metadata, *, fast=False, raw_depth=None):
    """Serialize one immutable RGB-D pair; never replace its acquisition time.

    Fast responses are generated per request, without ZIP compression or a
    serialized-response cache. The camera's latest frame is shared in memory.
    """
    if fast:
        source_h, source_w = depth.shape
        yy = np.arange(FAST_HEIGHT) * source_h // FAST_HEIGHT
        xx = np.arange(FAST_WIDTH) * source_w // FAST_WIDTH
        color = np.ascontiguousarray(color[yy[:, None], xx[None, :]])
        depth = np.ascontiguousarray(depth[yy[:, None], xx[None, :]], dtype=np.float32)
        metadata = deepcopy(metadata)
        intrinsic = metadata['color_intrinsic']
        sx, sy = FAST_WIDTH / source_w, FAST_HEIGHT / source_h
        metadata['color_intrinsic_original'] = deepcopy(intrinsic)
        metadata['color_intrinsic'] = dict(
            intrinsic, fx=intrinsic['fx'] * sx, fy=intrinsic['fy'] * sy,
            cx=intrinsic['cx'] * sx, cy=intrinsic['cy'] * sy,
            width=FAST_WIDTH, height=FAST_HEIGHT)
        metadata['source_image_size'] = {'width': source_w, 'height': source_h}
        metadata['resize_method'] = 'nearest'
        if 'aligned_depth_profile' in metadata:
            intrinsic = metadata['aligned_depth_profile']['intrinsic']
            metadata['aligned_depth_profile_original'] = deepcopy(metadata['aligned_depth_profile'])
            metadata['aligned_depth_profile']['intrinsic'] = dict(
                intrinsic, fx=intrinsic['fx'] * sx, fy=intrinsic['fy'] * sy,
                cx=intrinsic['cx'] * sx, cy=intrinsic['cy'] * sy,
                width=FAST_WIDTH, height=FAST_HEIGHT)
    buffer = io.BytesIO()
    save = np.savez if fast else np.savez_compressed
    arrays = dict(color_bgr=color, depth_mm=depth, metadata=np.array(json.dumps(metadata)))
    if raw_depth is not None and not fast:
        arrays['raw_depth_mm'] = raw_depth
    save(buffer, **arrays)
    return buffer.getvalue()

def preview_loop():
    """Render full-resolution monitor JPEGs at most 5 Hz, outside acquisition."""
    previous = None
    while True:
        start = time.monotonic()
        with lock:
            current = dict(snapshot) if state['camera'].get('ok') else {}
        if current and current['metadata']['capture']['time_unix'] != previous:
            try:
                color, depth = current['color'], current['depth']
                valid = depth > 0
                vis = cv2.applyColorMap(np.uint8(np.clip(depth/3000, 0, 1)*255), cv2.COLORMAP_TURBO)
                vis[~valid] = 0
                encoded = {}
                rectified=color_in_depth_grid(color,current['metadata'])
                for name, arr in [('color', color), ('color_rectified',rectified), ('depth', vis)]:
                    ok, jpg = cv2.imencode('.jpg', arr)
                    if not ok: raise RuntimeError('图像编码失败')
                    encoded[name] = jpg.tobytes()
                previous = current['metadata']['capture']['time_unix']
                stats = dict(depth_valid_percent=round(float(valid.mean()*100), 1),
                             depth_median_mm=round(float(np.median(depth[valid])), 1) if valid.any() else None,
                             preview_received_at=previous)
                with lock:
                    if state['camera'].get('ok'):
                        images.update(encoded)
                        state['camera'].update(stats)
                temp = live_dir / 'latest.tmp'
                temp.write_bytes(encoded['color'])
                temp.replace(live_dir / 'latest.jpg')
            except Exception as e:
                # A preview/disk failure must not stop RGB-D acquisition.
                print('Camera preview:', e, flush=True)
        time.sleep(max(0.001, PREVIEW_INTERVAL_S - (time.monotonic() - start)))

def camera_loop():
    while True:
        pipe = None
        started = False
        try:
            pipe = Pipeline()
            cfg = Config()
            for sensor in [OBSensorType.COLOR_SENSOR, OBSensorType.DEPTH_SENSOR]:
                profiles = pipe.get_stream_profile_list(sensor)
                if sensor == OBSensorType.COLOR_SENSOR:
                    try:
                        profile = profiles.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 8)
                    except Exception:
                        # The same Gemini335 currently advertises 10 fps rather
                        # than 8 fps. Preserve the calibrated image dimensions;
                        # obtain intrinsics from the actual selected profile.
                        profile = profiles.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 10)
                else:
                    profile = profiles.get_default_video_stream_profile()
                cfg.enable_stream(profile)
            cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            pipe.start(cfg)
            started = True
            align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            last = time.monotonic()
            frame_metadata = None
            frame_count = 0
            while True:
                frames = pipe.wait_for_frames(1000)
                # Host receipt of the original frameset, before alignment,
                # conversion or preview encoding. This is not exposure time.
                received_unix, received_monotonic = time.time(), time.monotonic()
                if frames is None:
                    if time.monotonic()-last > 5: raise RuntimeError('相机帧超时')
                    continue
                raw_c, raw_d = frames.get_color_frame(), frames.get_depth_frame()
                if raw_c is None or raw_d is None:
                    continue
                # Own a copy before AlignFilter; preserve the original depth
                # camera pixels and scale for offline reconstruction checks.
                raw_depth = np.frombuffer(raw_d.get_data(), dtype=np.uint16).reshape(
                    raw_d.get_height(), raw_d.get_width()).astype(np.float32)*raw_d.get_depth_scale()
                timing = {'color': frame_timing(raw_c), 'depth': frame_timing(raw_d)}
                raw_depth_profile = raw_d.get_stream_profile()
                raw_color_profile = raw_c.get_stream_profile()
                frames = align.process(frames)
                if frames is None:
                    if time.monotonic()-last > 5: raise RuntimeError('对齐彩色/深度帧超时')
                    continue
                c, d = frames.get_color_frame(), frames.get_depth_frame()
                if c is None or d is None:
                    if time.monotonic()-last > 5: raise RuntimeError('彩色或深度帧缺失')
                    continue
                data = np.asanyarray(c.get_data())
                fmt = c.get_format()
                if fmt == OBFormat.MJPG: color = cv2.imdecode(data, cv2.IMREAD_COLOR)
                elif fmt == OBFormat.RGB: color = cv2.cvtColor(data.reshape(c.get_height(),c.get_width(),3),cv2.COLOR_RGB2BGR)
                elif fmt == OBFormat.YUYV: color = cv2.cvtColor(data.reshape(c.get_height(),c.get_width(),2),cv2.COLOR_YUV2BGR_YUY2)
                elif fmt == OBFormat.BGR: color = data.reshape(c.get_height(),c.get_width(),3)
                else: raise RuntimeError(f'未支持的彩色格式 {fmt}')
                depth = np.frombuffer(d.get_data(), dtype=np.uint16).reshape(d.get_height(),d.get_width()).astype(np.float32)*d.get_depth_scale()
                if color is None or color.shape[:2] != depth.shape:
                    raise RuntimeError('彩色与对齐深度尺寸不一致')
                if frame_metadata is None:
                    profile = c.get_stream_profile().as_video_stream_profile()
                    intrinsic, distortion = profile.get_intrinsic(), profile.get_distortion()
                    device = pipe.get_device().get_device_info()
                    frame_metadata = {'camera': {'name': device.get_name(), 'serial': device.get_serial_number()}, 'color_intrinsic': {k:getattr(intrinsic,k) for k in ['fx','fy','cx','cy','width','height']}, 'color_distortion': {k:getattr(distortion,k) for k in ['k1','k2','p1','p2','k3','k4','k5','k6']}, 'color_distortion_model': str(distortion.model), 'color_profile_fps': profile.get_fps()}
                    ext = raw_depth_profile.get_extrinsic_to(raw_color_profile)
                    frame_metadata.update(
                        metadata_version=2, sdk_python_version=pyorbbecsdk.__version__,
                        camera_firmware=device.get_firmware_version(),
                        raw_depth_profile=profile_parameters(raw_depth_profile),
                        aligned_depth_profile=profile_parameters(d.get_stream_profile()),
                        raw_depth_to_color={'rotation_row_major': np.asarray(ext.rot).reshape(3, 3).tolist(),
                                            'translation_mm': np.asarray(ext.transform).reshape(3).tolist()},
                        align_filter_config_values={k: align.get_config_value(k) for k in
                            ['TargetDistortion', 'GapFillCopy', 'MatchTargetRes', 'AlignType']})
                frame_count += 1
                metadata = dict(frame_metadata, capture={
                    'time_unix': received_unix, 'received_monotonic': received_monotonic,
                    'timestamp_origin': 'host_frame_receive_unix',
                    'alignment': 'depth_to_color', 'depth_unit': 'mm',
                    'bridge_frame_counter': frame_count,
                    'frame_timing': timing,
                    'raw_depth_unit_mm': raw_d.get_depth_scale(),
                    'aligned_depth_unit_mm': d.get_depth_scale()})
                # Own the RGB buffer: some SDK formats expose reused memory.
                color = color.copy()
                value = dict(ok=True, received_at=received_unix, model='Orbbec Gemini 335', color_size=[c.get_width(),c.get_height()], depth_size=[d.get_width(),d.get_height()])
                with lock:
                    state['camera'].pop('error',None)
                    state['camera'].pop('color_only',None)
                    state['camera'].update(value)
                    snapshot.update(color=color, depth=depth, raw_depth=raw_depth, metadata=metadata)
                last = time.monotonic()
        except Exception as e:
            with lock:
                state['camera'] = dict(ok=False,error=str(e), received_at=time.time())
                snapshot.clear()
            print('Camera:', e, flush=True)
            # Limited UVC fallback: color only, release before the next SDK attempt.
            cap = cv2.VideoCapture('/dev/video6', cv2.CAP_V4L2)
            try:
                ok, frame = cap.read()
                if ok:
                    encoded_ok, jpg = cv2.imencode('.jpg', frame)
                    if encoded_ok:
                        with lock:
                            images['color'] = jpg.tobytes()
                            state['camera'].update(color_only=True, received_at=time.time(), color_size=[frame.shape[1],frame.shape[0]])
            finally:
                cap.release()
        finally:
            if started:
                try: pipe.stop()
                except Exception: pass
        time.sleep(2)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split('?')[0]
        if path == '/': body=(Path(__file__).parent/'web/monitor.html').read_bytes(); kind='text/html; charset=utf-8'
        elif path == '/status':
            with lock: body=json.dumps(state,ensure_ascii=False).encode()
            kind='application/json'
        elif path in ['/snapshot.npz', '/snapshot_fast.npz']:
            with lock:
                if not snapshot or not state['camera'].get('ok'):
                    self.send_error(503,'Waiting for camera'); return
                # Frame arrays/metadata are immutable after publication, so
                # references preserve their pairing without full-size copies.
                color, depth, metadata = snapshot['color'], snapshot['depth'], snapshot['metadata']
                raw_depth = snapshot.get('raw_depth')
            body=snapshot_packet(color, depth, metadata, fast=path == '/snapshot_fast.npz', raw_depth=raw_depth)
            kind='application/octet-stream'
        elif path in ['/color.jpg','/color_rectified.jpg','/depth.jpg']:
            with lock: body=images.get(path[1:-4])
            if body is None: self.send_error(503,'Waiting for camera'); return
            kind='image/jpeg'
        else: self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store, no-cache, max-age=0'); self.end_headers(); self.wfile.write(body)
    def log_message(self,*args): pass
if __name__ == '__main__':
    threading.Thread(target=robot_loop,daemon=True).start()
    threading.Thread(target=camera_loop,daemon=True).start()
    threading.Thread(target=preview_loop,daemon=True).start()
    print('Monitor: http://127.0.0.1:8765',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
