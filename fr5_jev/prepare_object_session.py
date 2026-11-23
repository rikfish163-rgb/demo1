"""Prepare a visible object's whole-task scope without device access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot_robot_fairino.action_control import finite_vector

from .durable_storage import NAME
from .object_geometry import measure_mask, selected_color_mask, side_routes, tilted_routes, top_down_routes
from .prepare_varied_session import fit_table, load_grasp_reference, load_scene
from .rgbd_geometry import color_to_depth_pixels
from .whole_task_collection import prepare_from_routes, task_instruction


def prepare(root, observation, object_description, rgb_pixel=None, *, mask_path=None, closing_percent,
            grasp_reference=None, grasp_offset=None, approach='top-down',
            grasp_below_top_mm=10., side_retraction_mm=0., tilt_from_down_deg=55.,
            tilted_closing_edge='long',
            split='training_complete_trial'):
    """Prepare one pick, five-second hold, release and full retreat for review."""
    task_instruction(5, 1, object_description)
    if type(closing_percent) is not int or not 0 <= closing_percent < 100:
        raise ValueError('Closing target must be an integer physical percentage below 100')
    if (grasp_reference is None) == (grasp_offset is None):
        raise ValueError('Provide exactly one observed grasp reference or explicit grasp offset')
    if not NAME.fullmatch(Path(root).name):
        raise ValueError('Recording name must be a simple filename')
    if approach not in ('top-down', 'side', 'tilted'):
        raise ValueError('Approach must be top-down, side or tilted')
    if approach != 'side' and side_retraction_mm != 0.:
        raise ValueError('Side retraction is only supported for a side approach')
    if approach != 'tilted' and tilt_from_down_deg != 55.:
        raise ValueError('Tilt angle is only supported for a tilted approach')
    if tilted_closing_edge not in ('long', 'short'):
        raise ValueError('Tilted closing edge must be long or short')
    if approach != 'tilted' and tilted_closing_edge != 'long':
        raise ValueError('Tilted closing edge is only supported for a tilted approach')
    if (rgb_pixel is None) == (mask_path is None):
        raise ValueError('Provide exactly one RGB pixel or PNG mask')

    if grasp_reference is not None:
        reference = load_grasp_reference(Path(grasp_reference))
        offset = reference['effective_grasp_offset_flange_mm']
    else:
        offset = finite_vector(grasp_offset, 3, 'grasp offset').tolist()
        reference = {'basis': 'explicit operator-supplied grasp offset',
                     'effective_grasp_offset_flange_mm': offset}

    observation = Path(observation).resolve()
    report, color, depth, calibration = load_scene(observation)
    if mask_path is None:
        pixel = finite_vector(rgb_pixel, 2, 'RGB pixel')
        mask = selected_color_mask(color, pixel)
        selection_method = 'color_pixel'
        selected_pixel = pixel.tolist()
    else:
        raw = Path(mask_path).read_bytes()
        if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Object mask must be a PNG image')
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError('Cannot decode the object mask PNG')
        if image.ndim == 2:
            mask = image != 0
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            mask = np.any(image[:, :, :3] != 0, axis=2)
            if image.shape[2] == 4:
                mask &= image[:, :, 3] != 0
        else:
            raise ValueError('Object mask must use grayscale or RGB pixels')
        if mask.shape != color.shape[:2]:
            raise ValueError('Object mask must use the current raw RGB pixel grid')
        v, u = np.nonzero(mask)
        if not len(u):
            raise ValueError('Object mask has no foreground pixels')
        center_u, center_v = u.mean(), v.mean()
        nearest = np.argmin((u - center_u) ** 2 + (v - center_v) ** 2)
        pixel = np.array([u[nearest], v[nearest]], dtype=float)
        selection_method = 'explicit_mask'
        selected_pixel = None
    depth_pixel = color_to_depth_pixels([pixel], report['camera'])[0]
    normal, table_offset, table = fit_table(color, depth, report, calibration, depth_pixel)
    measurement = measure_mask(mask, depth, report['camera'], calibration, normal, table_offset)
    start = finite_vector(report['robot_after']['tcp'], 6, 'observed start')
    route_builder = {'side': side_routes, 'tilted': tilted_routes, 'top-down': top_down_routes}[approach]
    route_options = {'penetration_mm': grasp_below_top_mm}
    if approach == 'side':
        route_options['side_retraction_mm'] = side_retraction_mm
    elif approach == 'tilted':
        route_options['tilt_from_down_deg'] = tilt_from_down_deg
        route_options['closing_edge'] = tilted_closing_edge
    routes, model = route_builder(start, measurement, offset, **route_options)
    geometry = {
        **model,
        'grasp_below_top_mm': model['penetration_mm'],
        'side_retraction_mm': model.get('side_retraction_mm', 0.),
        'source_observation': str(observation),
        'object_description': object_description,
        'closing_percent': closing_percent,
        'approach': approach,
        'selection_method': selection_method,
        'selection_mask': 'selection-mask.png',
        'selected_rgb_pixel': selected_pixel,
        'table_exclusion_rgb_pixel': pixel.tolist(),
        'selected_depth_pixel_for_table_exclusion': depth_pixel.tolist(),
        'table': table,
        'measurement': measurement,
        'grasp_reference': reference,
        'routes': routes,
    }
    if approach == 'tilted':
        geometry['tilt_from_down_deg'] = model['tilt_from_down_deg']
        geometry['tilted_closing_edge'] = model['closing_edge']
    result = prepare_from_routes(root, observation, object_description, routes, model['contact_tcp'],
                                 geometry, closing_percent=closing_percent, hold_seconds=5.,
                                 repetitions=1, retreat_mode='full', split=split, selection_mask=mask)
    return {key: value for key, value in result.items() if key != 'success'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True, help='Fresh recording spool directory')
    parser.add_argument('--observation', type=Path, required=True, help='Current read-only RGB-D observation')
    parser.add_argument('--object-description', required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--rgb-pixel', type=float, nargs=2, metavar=('U', 'V'))
    selection.add_argument('--mask', type=Path, help='PNG mask on the current raw RGB grid')
    parser.add_argument('--closing-percent', type=int, required=True)
    parser.add_argument('--approach', choices=('top-down', 'side', 'tilted'), default='top-down')
    parser.add_argument('--grasp-below-top-mm', type=float, default=10.,
                        help='Grasp center depth below the visible top in millimeters')
    parser.add_argument('--side-retraction-mm', type=float, default=0.,
                        help='Retract the side grasp center along the negative tool axis in millimeters')
    parser.add_argument('--tilt-from-down-deg', type=float, default=55.,
                        help='Tilt a tilted approach from the measured downward normal in degrees')
    parser.add_argument('--tilted-closing-edge', choices=('long', 'short'), default='long',
                        help='Visible object edge across which the tilted gripper closes')
    parser.add_argument('--split', choices=('training_complete_trial', 'held_out_complete_trial'),
                        default='training_complete_trial')
    grasp = parser.add_mutually_exclusive_group(required=True)
    grasp.add_argument('--grasp-reference', type=Path, help='Observed pick/place episode directory')
    grasp.add_argument('--grasp-offset', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                       help='Flange-frame grasp offset in millimeters')
    args = parser.parse_args()
    result = prepare(args.directory, args.observation, args.object_description, args.rgb_pixel,
                     closing_percent=args.closing_percent, grasp_reference=args.grasp_reference,
                     grasp_offset=args.grasp_offset, approach=args.approach, split=args.split,
                     grasp_below_top_mm=args.grasp_below_top_mm,
                     side_retraction_mm=args.side_retraction_mm, tilt_from_down_deg=args.tilt_from_down_deg,
                     tilted_closing_edge=args.tilted_closing_edge,
                     mask_path=args.mask)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
