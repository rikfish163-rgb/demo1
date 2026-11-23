"""Fit cup rim ellipses from outward image edges."""

import math

import cv2
import numpy as np

EDGE_SECTORS = 24
MIN_EDGE_SECTORS = 16


def _ellipse(gray, edges, gx, gy, x, y, radius, *, diagnostics=None):
    trace = {} if diagnostics is None else diagnostics

    def reject(reason):
        trace['rejected_by'] = reason
        return None

    x0, y0 = max(0, int(x-radius*1.22)), max(0, int(y-radius*1.22))
    x1, y1 = min(gray.shape[1], int(x+radius*1.22)+1), min(gray.shape[0], int(y+radius*1.22)+1)
    yy, xx = np.nonzero(edges[y0:y1, x0:x1]); xx, yy = xx+x0, yy+y0
    dx, dy = xx-x, yy-y
    radial = np.hypot(dx, dy)
    outward = (gx[yy,xx]*dx + gy[yy,xx]*dy)/np.maximum(radial, 1)
    keep = (radial > radius*.82) & (radial < radius*1.18) & (outward > 15)
    points = np.column_stack([xx[keep],yy[keep]]).astype(np.float32)
    trace['outward_edge_points'] = len(points)
    if len(points) < 35:
        return reject('edge_points')
    trace['fit_edge_points'] = []
    for _ in range(3):
        ellipse = cv2.fitEllipse(points)
        (cx,cy), (a,b), angle = ellipse
        if min(a,b) <= 0:
            return reject('ellipse_axes')
        theta = math.radians(angle)
        delta = points-np.array([cx,cy])
        X = delta[:,0]*math.cos(theta)+delta[:,1]*math.sin(theta)
        Y = -delta[:,0]*math.sin(theta)+delta[:,1]*math.cos(theta)
        residual = np.abs(np.sqrt((X/(a/2))**2+(Y/(b/2))**2)-1)
        chosen = points[residual < .065]
        trace['fit_edge_points'].append(len(chosen))
        if len(chosen) < 35:
            return reject('ellipse_residual')
        points = chosen
    (cx,cy), (a,b), angle = cv2.fitEllipse(points)
    trace.update(cup_uv=[float(cx),float(cy)], axes_px=[float(a),float(b)],
                 angle_degrees=float(angle), edge_points=len(points),
                 center_shift_px=float(np.hypot(cx-x,cy-y)))
    if (np.hypot(cx-x,cy-y) > radius*.22 or min(a,b) < radius*1.45
            or max(a,b) > radius*2.6 or max(a,b)/min(a,b) > 1.45):
        return reject('ellipse_shape')
    bins = np.floor((np.arctan2(points[:,1]-cy,points[:,0]-cx)+math.pi)/(2*math.pi)*EDGE_SECTORS).astype(int)%EDGE_SECTORS
    covered = len(np.unique(bins))
    coverage = covered/EDGE_SECTORS
    trace.update(edge_coverage=coverage, covered_sectors=covered,
                 total_sectors=EDGE_SECTORS, required_sectors=MIN_EDGE_SECTORS)
    if covered < MIN_EDGE_SECTORS:
        return reject('edge_coverage')
    trace['rejected_by'] = None
    return {'cup_uv':[float(cx),float(cy)], 'axes_px':[float(a),float(b)],
            'angle_degrees':float(angle), 'edge_coverage':coverage, 'edge_points':len(points)}


