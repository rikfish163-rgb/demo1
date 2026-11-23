"""Pure image profiles shared by FR5 stream training and inference.

``left_wrist_0_rgb`` is an existing model input slot. In the workspace
profile it contains a crop of the same front-camera frame, not wrist-camera
imagery. The policy's normal image preprocessor performs the later resize.
"""

from collections.abc import Mapping

import torch


FULL_IMAGE_PROFILE = 'full-v1'
WORKSPACE_IMAGE_PROFILE = 'full-and-workspace-v1'
WORKSPACE_ONLY_IMAGE_PROFILE = 'workspace-only-v1'
CUP_CENTERED_IMAGE_PROFILE = 'cup-centered-v1'
IMAGE_PROFILES = (FULL_IMAGE_PROFILE, WORKSPACE_IMAGE_PROFILE, WORKSPACE_ONLY_IMAGE_PROFILE,
                  CUP_CENTERED_IMAGE_PROFILE)
SOURCE_KEY = 'observation.images.front'
WORKSPACE_KEY = 'observation.images.left_wrist_0_rgb'
WORKSPACE_CROP_XYXY = (240, 100, 640, 360)  # half-open coordinates on a 640x360 frame
CUP_CROP_WIDTH, CUP_CROP_HEIGHT = 384, 256
CUP_CROP_ANCHOR_X, CUP_CROP_ANCHOR_Y = 230, 115


def image_profile_metadata(profile=FULL_IMAGE_PROFILE):
    """Return JSON-serializable provenance for a checkpoint's image profile."""
    if profile not in IMAGE_PROFILES:
        raise ValueError(f'Unknown image profile: {profile!r}')
    workspace = profile != FULL_IMAGE_PROFILE
    only = profile in (WORKSPACE_ONLY_IMAGE_PROFILE, CUP_CENTERED_IMAGE_PROFILE)
    info = {'schema': 'fr5-stream-image-profile-v1', 'name': profile,
            'source_key': SOURCE_KEY,
            'workspace_key': (SOURCE_KEY if only else WORKSPACE_KEY) if workspace else None,
            'workspace_crop_xyxy': list(WORKSPACE_CROP_XYXY) if workspace else None,
            'workspace_crop_coordinates': 'xyxy_half_open' if workspace else None,
            'workspace_view_source': 'crop_of_front_camera' if workspace else None,
            'workspace_is_physical_wrist_camera': False,
            'physical_camera_count': 1}
    if only:
        info['full_view_retained'] = False
        info['output_image_keys'] = [SOURCE_KEY]
    if profile == CUP_CENTERED_IMAGE_PROFILE:
        info.update(workspace_crop_xyxy=None, workspace_view_source='cup_centered_crop_of_front_camera',
                    crop_size_wh=[CUP_CROP_WIDTH, CUP_CROP_HEIGHT],
                    cup_anchor_crop_xy=[CUP_CROP_ANCHOR_X, CUP_CROP_ANCHOR_Y],
                    cup_uv_source_image_size_wh=[1920, 1080], source_image_size_wh=[640, 360],
                    edge_padding='replicate')
    return info


def apply_image_profile(batch, profile=FULL_IMAGE_PROFILE, *, initial_cup_uv=None):
    """Return a new batch mapping; never modify the input or its full image.

    The legacy full profile is an identity on values, including arbitrary
    pre-existing image layouts. The workspace profile accepts only a batched
    floating-point RGB tensor at 640x360 and refuses a pre-existing workspace
    key so it cannot silently overwrite or apply a crop twice. The derived
    tensor has independent storage. The explicit workspace-only profile replaces
    the front slot with that crop. The cup-centered profile offsets its crop
    using the opening rim pixels. Both keep just the front slot.
    """
    if profile not in IMAGE_PROFILES:
        raise ValueError(f'Unknown image profile: {profile!r}')
    if not isinstance(batch, Mapping):
        raise TypeError('Image profile batch must be a mapping')
    result = dict(batch)
    if profile == FULL_IMAGE_PROFILE:
        return result
    if WORKSPACE_KEY in batch:
        raise ValueError(f'Workspace image already exists: {WORKSPACE_KEY}')
    only = profile in (WORKSPACE_ONLY_IMAGE_PROFILE, CUP_CENTERED_IMAGE_PROFILE)
    if only:
        extras = [key for key in batch if key.startswith('observation.images.') and key != SOURCE_KEY]
        if extras:
            raise ValueError('Workspace-only input must contain only the original front image')
    if SOURCE_KEY not in batch:
        raise KeyError(f'Missing source image: {SOURCE_KEY}')
    image = batch[SOURCE_KEY]
    if not isinstance(image, torch.Tensor) or not image.is_floating_point():
        raise TypeError('Workspace source image must be a floating-point torch tensor')
    if image.ndim != 4 or tuple(image.shape[1:]) != (3, 360, 640):
        raise ValueError('Workspace source image must have shape [B, 3, 360, 640]')
    if profile == CUP_CENTERED_IMAGE_PROFILE:
        if not isinstance(initial_cup_uv, (list, tuple)) or len(initial_cup_uv) != image.shape[0]:
            raise ValueError('Cup-centered images require one opening cup position per image')
        from stream_observation_history import initial_cup_pixel
        crops = []
        for frame, uv in zip(image, initial_cup_uv):
            u, v = initial_cup_pixel(uv)
            left, top = round(u/3.)-CUP_CROP_ANCHOR_X, round(v/3.)-CUP_CROP_ANCHOR_Y
            ix = torch.arange(left, left+CUP_CROP_WIDTH, device=image.device).clamp_(0, 639)
            iy = torch.arange(top, top+CUP_CROP_HEIGHT, device=image.device).clamp_(0, 359)
            crops.append(frame.index_select(1, iy).index_select(2, ix))
        result[SOURCE_KEY] = torch.stack(crops)
        return result
    x0, y0, x1, y1 = WORKSPACE_CROP_XYXY
    key = SOURCE_KEY if only else WORKSPACE_KEY
    result[key] = image[:, :, y0:y1, x0:x1].clone()
    return result
