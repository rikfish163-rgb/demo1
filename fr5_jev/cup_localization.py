"""Read-only localization of this measured cup in the calibrated Gemini view.

Searches the full RGB image. The 75 mm upright rim/110 mm height are known
object dimensions, not a remembered pixel or a general-purpose cup detector.
Transparent surfaces can return table depth, so partial raised-wall evidence
is combined with the rim edge and appearance; ambiguity is reported, not hidden.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np

from .calibration import load_readonly_calibration
from .prepare_varied_session import fit_table, estimate_cup
from .rgbd_geometry import models, sample_rgb_points
from .cup_rim_geometry import EDGE_SECTORS, MIN_EDGE_SECTORS, _ellipse

SOURCE = 'calibrated_rim_and_depth_v1'


def detect(color, depth, meta, *, calibration=None, diagnostics=None):
    started = time.monotonic()
    result = {'source':SOURCE, 'status':'not_found', 'cup_uv':None, 'candidates':[],
              'message':'未找到清晰的杯口，可调整遮挡或使用手动修正',
              'camera_serial':meta.get('camera',{}).get('serial'),
              'capture_timestamp_unix':meta.get('capture',{}).get('time_unix'),
              'received_at':meta.get('capture',{}).get('time_unix'),
              'capture_timestamp_monotonic':meta.get('capture',{}).get('received_monotonic'),
              'known_cup_height_mm':110., 'known_rim_diameter_mm':75.,
              'search_region':'full_image', 'independent_llm_api_calls':0}
    if (color.dtype != np.uint8 or color.shape != (1080,1920,3) or depth.shape != (1080,1920)
            or result['camera_serial'] != 'CP1E54200015'):
        raise ValueError('杯位识别需要当前 Gemini 的完整 RGB-D 画面')
    K, _, _, _ = models(meta)
    if calibration is None:
        calibration = np.asarray(load_readonly_calibration(camera_serial=result['camera_serial'])['base_from_color_camera_mm'])
    report = {'camera':meta}
    # Exclude no fixed cup region: robust table fitting rejects raised objects.
    normal, offset, table = fit_table(color,depth,report,calibration,np.array([-10000.,-10000.]))
    result['table'] = table
    gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray,(5,5),1.1)
    edges = cv2.Canny(smooth,40,100)
    gx,gy = (cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3), cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3))
    # This object's apparent size follows the current table/calibration.
    _, rim, _ = estimate_cup(report,depth,calibration,np.array([K[0,2],K[1,2]]),normal,offset)
    camera_z = float((calibration[:3,:3].T@(rim-calibration[:3,3]))[2])
    expected_radius = float(np.sqrt(K[0,0]*K[1,1])*37.5/camera_z)
    traces = []
    if diagnostics is not None:
        diagnostics.update(expected_radius_px=expected_radius, candidate_traces=traces)
    circles = cv2.HoughCircles(smooth,cv2.HOUGH_GRADIENT,1.2,max(30,expected_radius),
        param1=90,param2=28,minRadius=max(12,int(expected_radius*.72)),maxRadius=int(expected_radius*1.32))
    result['circle_candidates'] = 0 if circles is None else len(circles[0])
    candidates = []
    if circles is not None:
        for index,(x,y,radius) in enumerate(circles[0]):
            trace = {'index':index, 'hough_uvr':[float(x),float(y),float(radius)],
                     'ellipse':{}, 'rejected_by':None}
            if diagnostics is not None:
                traces.append(trace)
            ellipse = _ellipse(gray,edges,gx,gy,float(x),float(y),float(radius),
                               diagnostics=trace['ellipse'])
            if ellipse is None:
                trace['rejected_by'] = 'ellipse.'+trace['ellipse']['rejected_by']
                continue
            x,y = ellipse['cup_uv']; a,b = ellipse['axes_px']; radius = math.sqrt(a*b)/2
            if min(x,y) < radius*1.4 or x+radius*1.4 >=1920 or y+radius*1.4 >=1080:
                trace['rejected_by'] = 'image_border'
                continue
            _, rim, _ = estimate_cup(report,depth,calibration,np.array([x,y]),normal,offset)
            predicted_z = float((calibration[:3,:3].T@(rim-calibration[:3,3]))[2])
            expected = np.sqrt(K[0,0]*K[1,1])*37.5/predicted_z
            ratio = radius/expected
            trace.update(radius_px=float(radius), expected_radius_px=float(expected),
                         size_ratio=float(ratio))
            if not .78 < ratio < 1.25:
                trace['rejected_by'] = 'physical_size'
                continue
            yy,xx = np.mgrid[int(y-radius*1.4):int(y+radius*1.4)+1,
                             int(x-radius*1.4):int(x+radius*1.4)+1]
            distance = np.hypot(xx-x,yy-y)/radius
            inner = gray[yy[distance<.75],xx[distance<.75]]
            ring = (distance>1.15)&(distance<1.35)
            outer = gray[yy[ring],xx[ring]]
            contrast = float(np.median(outer))-float(np.median(inner))
            trace['interior_contrast'] = contrast
            if contrast < 20:
                trace['rejected_by'] = 'interior_contrast'
                continue
            angles = np.linspace(0,2*np.pi,64,endpoint=False)
            uv = np.column_stack([x+.9*radius*np.cos(angles), y+.9*radius*np.sin(angles)])
            sampled = sample_rgb_points(uv,depth,meta,radius=1,maximum_patch_spread_mm=1000)
            pts = sampled['points_camera_mm']; pts = pts[np.isfinite(pts).all(axis=1)]
            trace['valid_depth_samples'] = len(pts)
            if len(pts) < 16:
                trace['rejected_by'] = 'valid_depth_samples'
                continue
            heights = (pts@calibration[:3,:3].T+calibration[:3,3])@normal-offset
            near_rim = float(np.mean((heights>65)&(heights<140)))
            p90 = float(np.percentile(heights,90))
            trace.update(raised_wall_fraction=near_rim,
                         height_percentiles_mm=np.percentile(heights,[10,50,90]).tolist())
            if near_rim < .15 or not 60 < p90 < 170:
                trace['rejected_by'] = 'raised_wall_depth'
                continue
            ellipse.update(radius_px=radius, expected_radius_px=float(expected),
                size_ratio=float(ratio), interior_contrast=contrast,
                raised_wall_fraction=near_rim, height_percentiles_mm=np.percentile(heights,[10,50,90]).tolist(),
                score=float(ellipse['edge_coverage']+min(contrast/100,1)+near_rim-abs(math.log(ratio))))
            if not any(np.linalg.norm(np.asarray(c['cup_uv'])-[x,y]) < radius*.65 for c in candidates):
                candidates.append(ellipse)
                trace['accepted'] = True
            else:
                trace['rejected_by'] = 'duplicate_candidate'
    candidates.sort(key=lambda item:item['score'],reverse=True)
    result['candidates'] = candidates
    if len(candidates)==1:
        result.update(status='found',cup_uv=candidates[0]['cup_uv'],message='已自动定位杯口')
    elif len(candidates)>1:
        result.update(status='ambiguous',message='发现多个相似杯口，可用手动修正选择目标')
    result['processing_seconds'] = time.monotonic()-started
    return result


def snapshot():
    raw = urllib.request.urlopen('http://127.0.0.1:8765/snapshot.npz',timeout=5).read()
    with np.load(io.BytesIO(raw),allow_pickle=False) as packet:
        color,depth = packet['color_bgr'],packet['depth_mm']
        meta = json.loads(str(packet['metadata'].item()))
    age = time.monotonic()-meta['capture']['received_monotonic']
    if not 0 <= age < 2:
        raise RuntimeError('相机画面尚未更新')
    result = detect(color,depth,meta)
    result['received_at'] = meta['capture']['time_unix']
    return result


def from_observation(directory):
    from .prepare_varied_session import load_scene
    report,color,depth,calibration = load_scene(Path(directory))
    return detect(color,depth,report['camera'],calibration=calibration)
