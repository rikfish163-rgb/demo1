"""Explicit RGB/depth pixel geometry. No device access or metric calibration claim."""
from functools import lru_cache

import cv2
import numpy as np


def optics(intrinsic, distortion):
    K=np.array([[intrinsic['fx'],0,intrinsic['cx']],
                [0,intrinsic['fy'],intrinsic['cy']],[0,0,1.]],float)
    D=np.array([distortion[k] for k in ('k1','k2','p1','p2','k3','k4','k5','k6')],float)
    if not np.isfinite(K).all() or not np.isfinite(D).all() or min(K[0,0],K[1,1])<=0:
        raise ValueError('Invalid camera parameters')
    return K,D


def models(meta):
    if meta.get('capture',{}).get('alignment')!='depth_to_color' or meta['capture'].get('depth_unit')!='mm':
        raise ValueError('Expected depth aligned into color camera frame, in mm')
    profile=meta.get('aligned_depth_profile')
    if profile is None:
        raise ValueError('Actual aligned-depth pixel model missing; legacy snapshots need an explicit offline audit')
    if 'BROWN_CONRADY' not in profile.get('distortion_model','') or 'BROWN_CONRADY' not in meta.get('color_distortion_model',''):
        raise ValueError('Unsupported distortion model')
    return (*optics(meta['color_intrinsic'],meta['color_distortion']),
            *optics(profile['intrinsic'],profile['distortion']))


def _rays(pixels,K,D):
    pixels=np.asarray(pixels,float).reshape(-1,2)
    if not np.isfinite(pixels).all():raise ValueError('Nonfinite pixel coordinates')
    xy=cv2.undistortPoints(pixels.reshape(-1,1,2),K,D).reshape(-1,2)
    return np.c_[xy,np.ones(len(xy))]


def color_to_depth_pixels(pixels,meta):
    K,D,Kd,Dd=models(meta)
    return cv2.projectPoints(_rays(pixels,K,D),np.zeros(3),np.zeros(3),Kd,Dd)[0].reshape(-1,2)


def depth_pixels_to_camera_points(pixels,z_mm,meta):
    _,_,K,D=models(meta)
    z=np.asarray(z_mm,float).reshape(-1)
    directions=_rays(pixels,K,D)
    if len(z)!=len(directions):raise ValueError('Pixel/depth count mismatch')
    points=directions*z[:,None]
    points[~np.isfinite(z)|(z<=0)]=np.nan
    return points


def sample_rgb_points(pixels,depth,meta,*,radius=2,maximum_patch_spread_mm=8.):
    """Map RGB pixels before sampling; holes, borders and depth edges stay invalid."""
    mapped=color_to_depth_pixels(pixels,meta)
    intr=meta['aligned_depth_profile']['intrinsic']
    if depth.shape!=(intr['height'],intr['width']):raise ValueError('Depth image/model size mismatch')
    if radius<0:raise ValueError('Invalid patch radius')
    z=np.full(len(mapped),np.nan);spread=np.full(len(mapped),np.nan)
    required=max(1,int(np.ceil(.6*(2*radius+1)**2)))
    for i,(u,v) in enumerate(np.rint(mapped).astype(int)):
        if not (radius<=u<depth.shape[1]-radius and radius<=v<depth.shape[0]-radius):continue
        patch=depth[v-radius:v+radius+1,u-radius:u+radius+1]
        good=patch[np.isfinite(patch)&(patch>0)]
        if len(good)<required:continue
        spread[i]=float(np.ptp(good))
        if spread[i]<=maximum_patch_spread_mm:z[i]=float(np.median(good))
    # Both grids are in the color camera frame; the ray comes from the
    # original RGB pixel, not the nearest integer depth pixel.
    K,D,_,_=models(meta)
    points=_rays(pixels,K,D)*z[:,None]
    return {'points_camera_mm':points,'depth_pixels':mapped,'depth_mm':z,
            'patch_spread_mm':spread,'valid':np.isfinite(z)}


@lru_cache(maxsize=4)
def _color_remap(parameters):
    width,height,*values=parameters
    K=np.array(values[:9]).reshape(3,3);D=np.array(values[9:17])
    Kd=np.array(values[17:26]).reshape(3,3);Dd=np.array(values[26:34])
    if np.all(Dd==0):
        return cv2.initUndistortRectifyMap(K,D,np.eye(3),Kd,(width,height),cv2.CV_32FC1)
    yy,xx=np.mgrid[:height,:width]
    directions=_rays(np.c_[xx.ravel(),yy.ravel()],Kd,Dd)
    uv=cv2.projectPoints(directions,np.zeros(3),np.zeros(3),K,D)[0].reshape(height,width,2)
    return uv[:,:,0].astype(np.float32),uv[:,:,1].astype(np.float32)


def color_in_depth_grid(color,meta):
    K,D,Kd,Dd=models(meta)
    rgb=meta['color_intrinsic'];aligned=meta['aligned_depth_profile']['intrinsic']
    if color.shape[:2]!=(rgb['height'],rgb['width']):raise ValueError('RGB image/model size mismatch')
    key=(int(aligned['width']),int(aligned['height']),*K.ravel(),*D,*Kd.ravel(),*Dd)
    mapx,mapy=_color_remap(key)
    return cv2.remap(color,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
