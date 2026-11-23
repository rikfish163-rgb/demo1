"""Select recorded state history at fixed offsets from the current time."""

from bisect import bisect_right
import math
import re


HISTORY_OFFSETS_S = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0)
NONE_PROFILE = 'none'
HISTORY_PROFILE = 'state-history-v1'
PHYSICAL_HISTORY_PROFILE = 'state-history-physical-v1'
CUP_HISTORY_PROFILE = 'state-history-physical-initial-cup-v1'
CUP_TCP_HISTORY_PROFILE = 'state-history-physical-cup-tcp-v1'
CUP_TRACKING_PROFILE = 'state-history-physical-cup-tracking-v1'
CUP_TIMING_PROFILE = 'state-history-physical-cup-timing-v1'
CUP_ACTION_CLOCK_PROFILE = 'state-history-physical-cup-action-clock-v1'
CUP_ALIGNED_PROFILE = 'state-history-physical-cup-aligned-v1'
CUP_SETDOWN_PROFILE = 'state-history-physical-cup-setdown-v1'
CUP_SETDOWN_CLEAN_PROFILE = 'state-history-physical-cup-setdown-clean-v1'
CUP_SETDOWN_FIXED_PROFILE = 'state-history-physical-cup-setdown-fixed-v1'
CUP_SETDOWN_PROFILES = (CUP_SETDOWN_PROFILE, CUP_SETDOWN_CLEAN_PROFILE, CUP_SETDOWN_FIXED_PROFILE)
CUP_CAMERA_CLOCK_PROFILES = (CUP_ALIGNED_PROFILE, *CUP_SETDOWN_PROFILES)
CUP_ACTION_PROFILES = (CUP_ACTION_CLOCK_PROFILE, *CUP_CAMERA_CLOCK_PROFILES)
CUP_TIMING_PROFILES = (CUP_TIMING_PROFILE, *CUP_ACTION_PROFILES)
CUP_TRACKING_PROFILES = (CUP_TRACKING_PROFILE, *CUP_TIMING_PROFILES)
CUP_TCP_HISTORY_PROFILES = (CUP_TCP_HISTORY_PROFILE, *CUP_TRACKING_PROFILES)
CUP_HISTORY_PROFILES = (CUP_HISTORY_PROFILE, *CUP_TCP_HISTORY_PROFILES)
PHYSICAL_HISTORY_PROFILES = (PHYSICAL_HISTORY_PROFILE, *CUP_HISTORY_PROFILES)
HISTORY_PROFILES = (HISTORY_PROFILE, *PHYSICAL_HISTORY_PROFILES)
PROFILES = (NONE_PROFILE, *HISTORY_PROFILES)
CUP_IMAGE_WIDTH, CUP_IMAGE_HEIGHT = 1920, 1080
CUP_PIXEL_BIN = 8
CUP_TRACKING_BIN_MM = 5
CUP_TIMING_PHASES = ('approaching', 'at_pickup', 'grasp_contact', 'lifting', 'hover',
                     'returning', 'at_table_height', 'released', 'unknown')
CUP_TIMING_VISION = ('table_height', 'raised', 'uncertain_height', 'missing', 'disagrees')


def context_metadata(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown observation context profile')
    result = {'schema': 'fr5-stream-observation-context-v1', 'profile': profile,
            'offsets_s': list(HISTORY_OFFSETS_S), 'age_unit': 'seconds',
            'age_decimal_places': 3, 'joint_unit': 'milliradians',
            'gripper_unit': 'integer_percent', 'valid_unit': '0_or_1',
            'training_clock': ('host_monotonic_camera_time' if profile in PHYSICAL_HISTORY_PROFILES
                               else 'retimed_episode_task_clock'),
            'runtime_clock': 'host_monotonic_camera_time'}
    if profile in CUP_HISTORY_PROFILES:
        result['initial_cup'] = {'meaning': 'rim center before the first robot action; fixed for the episode',
                                 'image_size_wh': [CUP_IMAGE_WIDTH, CUP_IMAGE_HEIGHT],
                                 'pixel_bin': CUP_PIXEL_BIN, 'camera_serial': 'CP1E54200015'}
    if profile in CUP_TCP_HISTORY_PROFILES:
        result.update(joint_unit=None, pose_units=['meters_from_cup_base']*3+['base_radians']*3,
                      pose_anchor='initial_cup_base_in_J1', measured_pose='tcp_tool0')
    if profile in CUP_TRACKING_PROFILES:
        result['cup_tracking'] = {
            'coordinates': 'J1 rim center minus nominal PGI-140 jaw tip center',
            'bin_mm': CUP_TRACKING_BIN_MM, 'observation': 'synchronized Gemini RGB-D and FR5 flange',
            'missing': 'no position without fresh rim evidence',
            'contact': 'native settled contact bit; unknown when gripper feedback invalid',
        }
    if profile in CUP_TIMING_PROFILES:
        result['cup_timing'] = {'states': list(CUP_TIMING_PHASES), 'vision_states': list(CUP_TIMING_VISION),
                                'hover_clock': 'observed continuous steady airborne flange, host monotonic',
                                'hover_text_resolution_s': .5, 'future_data_used': False}
    if profile == CUP_ACTION_CLOCK_PROFILE:
        result['action_clock'] = {'input': 'padded pi0.5 action channels at each denoising step',
                                  'fields': ['remaining_hover_time', 'clock_known', 'phase', 'rim_evidence'],
                                  'tick_hz': 6, 'scale': 3.0}
    if profile in CUP_CAMERA_CLOCK_PROFILES:
        result['action_clock'] = {
            'input': 'padded pi0.5 action channels at each denoising step', 'scale': 3., 'tick_hz': 6,
            'fields': ['remaining_hover_time', 'clock_known', 'phase', 'rim_evidence',
                       'rim_minus_jaw_xyz', 'rim_known', 'native_contact', 'contact_known',
                       'support_gap_mm', 'support_gap_known'],
            'action_targets': 'camera_time_plus_tick',
        }
        if profile in CUP_SETDOWN_PROFILES:
            result['action_clock']['fields'].append('setdown_age_s')
            result['cup_timing']['setdown'] = {
                'meaning': 'elapsed at measured grasp height on return with closed contact and stationary TCP',
                'visual': 'a conflicting rim height resets elapsed time; occlusion is allowed',
                'height_tolerance_mm': 1.5, 'tcp_speed_limit_mm_s': 4.,
                'max_observation_gap_s': .35, 'scale_s': 2.,
            }
        if profile in (CUP_SETDOWN_CLEAN_PROFILE, CUP_SETDOWN_FIXED_PROFILE):
            result['action_clock']['input'] = 'fixed conditioning in padded pi0.5 action channels at each denoising step'
        if profile == CUP_SETDOWN_CLEAN_PROFILE:
            result['cup_timing']['setdown']['scale_s'] = .30
    return result


def initial_cup_pixel(uv):
    if (not isinstance(uv, list) or len(uv) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x < bound
                   for x, bound in zip(uv, (CUP_IMAGE_WIDTH, CUP_IMAGE_HEIGHT)))):
        raise ValueError('Expected initial cup rim center in full Gemini RGB pixels')
    return [float(x) for x in uv]


def cup_tracking_input(value):
    if not isinstance(value, dict) or set(value) != {'rim_minus_jaw_mm', 'native_contact'}:
        raise ValueError('Expected a cup tracking observation with an explicit missing value')
    delta = value['rim_minus_jaw_mm']
    if delta is not None and (not isinstance(delta, list) or len(delta) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in delta)):
        raise ValueError('Expected finite tracked cup-to-jaw XYZ in millimetres')
    contact = value['native_contact']
    if contact is not None and (type(contact) is not int or contact not in (0, 1)):
        raise ValueError('Expected native gripper contact or unknown')
    return value


def cup_timing_input(value):
    base = {'phase', 'hover_seconds', 'visual'}
    if (not isinstance(value, dict) or set(value) not in (base, base | {'support_gap_mm'},
                                                          base | {'support_gap_mm', 'setdown_age_s'})
            or value['phase'] not in CUP_TIMING_PHASES or value['visual'] not in CUP_TIMING_VISION):
        raise ValueError('Expected a measured cup timing state')
    seconds = value['hover_seconds']
    if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('Expected measured nonnegative hover seconds or unknown')
    gap = value.get('support_gap_mm')
    if gap is not None and (type(gap) not in (int, float) or not math.isfinite(gap)):
        raise ValueError('Expected measured height above grasp or unknown')
    age = value.get('setdown_age_s')
    if age is not None and (type(age) not in (int, float) or not math.isfinite(age) or age < 0.
                            or gap is None or value['phase'] not in ('returning', 'at_table_height')):
        raise ValueError('Expected causal setdown age or unknown')
    return value


def action_clock(task, cup_timing, *, cup_tracking=None, profile=CUP_ACTION_CLOCK_PROFILE):
    timing = cup_timing_input(cup_timing)
    requested = set(float(x) for x in re.findall(r'在空中维持\s*([0-9]+(?:\.[0-9]+)?)\s*秒', task))
    elapsed = timing['hover_seconds']
    duration = next(iter(requested)) if len(requested) == 1 else 0.
    active = 0 < duration < math.inf and elapsed is not None and timing['phase'] == 'hover'
    result = [duration - elapsed if active else 0., float(active),
              *(float(timing['phase'] == p) for p in CUP_TIMING_PHASES),
              *(float(timing['visual'] == v) for v in CUP_TIMING_VISION)]
    if profile in CUP_CAMERA_CLOCK_PROFILES:
        track = cup_tracking_input(cup_tracking)
        rim, contact, gap = track['rim_minus_jaw_mm'], track['native_contact'], timing['support_gap_mm']
        bound = lambda x: max(-1., min(1., x))
        result += [*(bound(v / 60.) for v in (rim or [0., 0., 0.])), float(rim is not None),
                   float(contact or 0), float(contact is not None),
                   bound(gap / 12.) if gap is not None else 0., float(gap is not None)]
        if profile in CUP_SETDOWN_PROFILES:
            age = timing['setdown_age_s']
            if profile == CUP_SETDOWN_CLEAN_PROFILE:
                result.append(min(1., age / .30) if age is not None else 0.)
            else:
                result.append(min(1., (age + .15) / 2.) if age is not None else 0.)
    return result


def render_task(task, history, profile=NONE_PROFILE, *, initial_cup_uv=None, cup_tracking=None, cup_timing=None):
    """Render only model input text; keep the caller's task and history intact."""
    if profile == NONE_PROFILE:
        return task
    if profile not in HISTORY_PROFILES:
        raise ValueError('Unknown observation context profile')
    if not isinstance(history, list) or len(history) != len(HISTORY_OFFSETS_S):
        raise ValueError('observation_history must contain six entries')
    rendered = []
    for offset, item in zip(HISTORY_OFFSETS_S, history):
        if item is None:
            rendered.append('-')
            continue
        if not isinstance(item, dict) or set(item) != {'age_s', 'state'}:
            raise ValueError('Invalid observation history entry')
        age, state = item['age_s'], item['state']
        if (type(age) not in (int, float) or not math.isfinite(age) or age < offset - 1e-6
                or not isinstance(state, list) or len(state) != 8
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in state)
                or not 0 <= state[6] <= 1 or state[7] not in (0, 1)):
            raise ValueError('Invalid observation history age or state')
        pose = ','.join(str(round(x * 1000)) for x in state[:6])
        age_text = f'{age:.3f}'.rstrip('0').rstrip('.')
        rendered.append(f'{age_text}:{pose},{round(state[6] * 100)},{int(state[7])}')
    if profile in CUP_HISTORY_PROFILES:
        u, v = (int(x // CUP_PIXEL_BIN) for x in initial_cup_pixel(initial_cup_uv))
        task += f' Initial cup rim(u,v in 8px bins): {u},{v}.'
    if profile in CUP_TRACKING_PROFILES:
        tracking = cup_tracking_input(cup_tracking)
        delta, contact = tracking['rim_minus_jaw_mm'], tracking['native_contact']
        relative = '-' if delta is None else ','.join(str(round(x / CUP_TRACKING_BIN_MM)) for x in delta)
        contact_text = '-' if contact is None else str(contact)
        task += f' Cup rim from jaw (xyz in {CUP_TRACKING_BIN_MM}mm bins): {relative}; contact: {contact_text}.'
    if profile in CUP_TIMING_PROFILES:
        timing = cup_timing_input(cup_timing)
        seconds = timing['hover_seconds']
        hover = '-' if seconds is None else f'{math.floor(seconds*2)/2:g}'
        task += f" Cup state: {timing['phase']}; hover s: {hover}; rim: {timing['visual']}."
    pose_units = 'tcp_xyz_mm_from_cup,rpy_mrad' if profile in CUP_TCP_HISTORY_PROFILES else 'joint_mrad'
    return task + f' History(age_s:{pose_units},grip_pct,valid; - missing): ' + ';'.join(rendered)


def select_observation_history(samples, current_time):
    """Select the latest state at or before each requested historical time.

    Samples are ordered ``(timestamp, state8)`` pairs from one episode and
    use the same clock as ``current_time``. Return a mapping from requested
    offset in seconds to either None or the recorded timestamp, actual age,
    and an unchanged copy of the eight-dimensional state. A dropped frame can
    make the actual age greater than the requested offset.
    """
    timestamps = []
    states = []
    for timestamp, state in samples:
        if timestamps and timestamp < timestamps[-1]:
            raise ValueError('Samples must be ordered by timestamp')
        if len(state) != 8:
            raise ValueError('Each sample must have eight state values')
        timestamps.append(timestamp)
        states.append(state)

    result = {}
    for offset in HISTORY_OFFSETS_S:
        index = bisect_right(timestamps, current_time - offset) - 1
        if index < 0:
            result[offset] = None
        else:
            timestamp = timestamps[index]
            result[offset] = {'timestamp': timestamp, 'age_s': current_time - timestamp,
                              'state': list(states[index])}
    return result
