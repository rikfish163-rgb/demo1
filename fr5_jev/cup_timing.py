"""Causal cup timing observations from synchronized Gemini and J1 feedback.

The flange records displacement, not proof that the cup remained in the jaws.
Fresh rim evidence distinguishes agreement, occlusion, and disagreement with
that estimate. Only the policy proposes motion and gripper targets.
"""
from collections import deque
import math

import numpy as np


# These distances describe the recorded cup, PGI-140 and 50 mm lift envelope.
PICKUP_FLANGE_FROM_CUP_MM = np.array([159.7, 71.3, 95.8])
RIM_FROM_JAW_AT_CONTACT_MM = np.array([-13.5, -3.4, 13.9])
_LIFT_MM = 7.
_HOVER_MIN_MM = 28.
_TABLE_BAND_MM = 3.
_STILL_SECONDS = .5
_STILL_DISTANCE_MM = 2.
_STILL_SPEED_MM_S = 4.
_MAX_FRAME_GAP_S = .55
SETDOWN_HEIGHT_MM = 1.5
SETDOWN_SPEED_MM_S = 4.
SETDOWN_MIN_AGE_S = .30
SETDOWN_GAP_S = .35


class CupTiming:
    def __init__(self, initial_cup, *, aligned=False, setdown=False):
        self.normal = np.asarray(initial_cup['table_normal'], float)
        self.initial_rim = np.asarray(initial_cup['rim_xyz_mm'], float)
        self.cup_base = np.asarray(initial_cup['base_xyz_mm'], float)
        self.serial = initial_cup['camera_serial']
        self.time = None
        self.samples = deque()
        self.contact_time = None
        self.contact_flange = None
        self.was_lifted = False
        self.was_released = False
        self.hover_start = None
        self.hover_seconds = None
        self.was_returning = False
        self.disagree_since = None
        self.aligned = aligned or setdown
        self.setdown = setdown
        self.setdown_since = None
        self.setdown_last = None
        self.retreat_peak = None
        self.hover_level = None
        self.descent_start = None

    def _clear_contact(self):
        self.contact_time = self.contact_flange = self.hover_start = self.hover_seconds = None
        self.was_lifted = self.was_returning = False
        self.disagree_since = None
        self.hover_level = self.descent_start = None
        self.setdown_since = self.setdown_last = None

    def update(self, camera, robot, tracking):
        """Consume one real capture, without action targets or future frames."""
        stamp = float(camera['source_timestamp_monotonic'])
        feedback_stamp = float(robot['received_monotonic'])
        flange = np.asarray(robot['flange'][:3], float)
        speed = abs(float(robot['tcp_speed'][0]))
        if (camera['source_camera_serial'] != self.serial or not math.isfinite(stamp)
                or not math.isfinite(feedback_stamp) or abs(stamp-feedback_stamp) > .1
                or not np.isfinite(flange).all() or not math.isfinite(speed)
                or robot['tool'] != 0 or robot['user'] != 0
                or self.time is not None and stamp <= self.time):
            raise ValueError('Cup timing requires strictly ordered paired Gemini and J1 feedback')
        if self.time is not None and stamp - self.time > _MAX_FRAME_GAP_S:
            self._clear_contact()
            self.samples.clear()
            if self.aligned:
                self.was_released, self.retreat_peak = False, None
        self.time = stamp
        contact = tracking['native_contact']
        rim = tracking['rim_xyz_mm']
        rim_rise = None if rim is None else float((np.asarray(rim) - self.initial_rim) @ self.normal)
        visual = 'missing' if rim_rise is None else 'table_height' if abs(rim_rise) <= 9. else (
            'raised' if rim_rise >= 16. else 'uncertain_height')
        flange_error = flange - self.cup_base - PICKUP_FLANGE_FROM_CUP_MM
        pickup_distance = float(np.linalg.norm(flange_error))
        at_pickup = np.linalg.norm(flange_error[:2]) <= 15. and abs(flange_error[2]) <= 10.
        # If visible, a rim well outside the observed finger/cup relation
        # should not convert arbitrary PGI contact into cup contact.
        relative = tracking['rim_minus_jaw_mm']
        if relative is not None and np.linalg.norm(np.asarray(relative) - RIM_FROM_JAW_AT_CONTACT_MM) > 38.:
            at_pickup = False

        self.samples.append((stamp, flange, contact, speed))
        while len(self.samples) > 1 and self.samples[1][0] <= stamp - _STILL_SECONDS:
            self.samples.popleft()
        if contact == 0:
            if self.contact_flange is not None:
                self.was_released = robot['gripper_position'] >= 95
                self.retreat_peak = pickup_distance if self.was_released else None
                self._clear_contact()
            if self.aligned and self.was_released:
                self.retreat_peak = max(self.retreat_peak, pickup_distance)
                if self.retreat_peak - pickup_distance >= 6.:
                    self.was_released, self.retreat_peak = False, None
        elif contact == 1 and self.contact_flange is None and at_pickup and visual != 'raised':
            self.contact_time, self.contact_flange = stamp, flange
            self.was_released = False

        height = None
        if contact == 0:
            phase = ('released' if self.was_released else 'at_pickup' if at_pickup else 'approaching')
        elif contact is None or self.contact_flange is None:
            phase = 'unknown'
        else:
            height = float((flange-self.contact_flange) @ self.normal)
            self.was_lifted |= height >= _LIFT_MM
            first, _, _, _ = self.samples[0]
            steady = (stamp-first >= _STILL_SECONDS and first >= self.contact_time
                      and all(c == 1 and np.linalg.norm(p-flange) <= _STILL_DISTANCE_MM
                              for _, p, c, _ in self.samples))
            if (rim_rise is not None and abs(rim_rise-height) > 23.
                    and (visual == 'table_height' and height >= _HOVER_MIN_MM
                         or visual == 'raised' and height <= _TABLE_BAND_MM)):
                if self.disagree_since is None:
                    self.disagree_since = stamp
            else:
                self.disagree_since = None
            if self.disagree_since is not None and stamp-self.disagree_since >= .35:
                visual = 'disagrees'
            if self.hover_start is not None:
                if self.aligned:
                    # Freeze at the first measured drop from the stable level,
                    # after a second sample confirms that descent is underway.
                    drop = self.hover_level - height
                    if not self.was_returning:
                        if drop > 1.:
                            if self.descent_start is None:
                                self.descent_start = stamp
                            elif drop > 2.:
                                self.was_returning = True
                                self.hover_seconds = self.descent_start - self.hover_start
                        else:
                            self.descent_start = None
                    if not self.was_returning and not steady and self.descent_start is None:
                        self.hover_start = self.hover_seconds = self.hover_level = None
                elif not steady:
                    self.was_returning = True
            if not self.was_lifted:
                phase = 'grasp_contact'
            elif (height >= _HOVER_MIN_MM and not self.was_returning
                  and (steady or self.aligned and self.descent_start is not None)):
                if self.hover_start is None:
                    self.hover_start = first
                    if self.aligned:
                        self.hover_start = next((t for t, p, c, v in self.samples if c == 1 and
                                                 v <= _STILL_SPEED_MM_S and
                                                 np.linalg.norm(p-flange) <= 1.), stamp)
                        self.hover_level = height
                self.hover_seconds = (self.descent_start if self.aligned and self.descent_start is not None
                                      else stamp) - self.hover_start
                phase = 'hover'
            elif self.hover_start is not None:
                phase = 'returning'
            else:
                phase = 'lifting'
            if (self.was_lifted and self.hover_start is not None
                    and abs(height) <= (1.5 if self.aligned else _TABLE_BAND_MM)
                    and steady and visual not in ('raised', 'disagrees')):
                phase = 'at_table_height'
            if visual == 'disagrees':
                phase = 'unknown'

        state = {'phase': phase, 'hover_seconds': self.hover_seconds, 'visual': visual}
        if self.aligned:
            state['support_gap_mm'] = height if contact == 1 and phase != 'unknown' else None
        if self.setdown:
            # A stationary wrist during the airborne hold is never a setdown.
            # The rim can be occluded; a visible rim at a conflicting height
            # is evidence against support, not a missing measurement.
            near = (contact == 1 and phase in ('returning', 'at_table_height')
                    and abs(height) <= SETDOWN_HEIGHT_MM and visual in ('table_height', 'missing')
                    and abs(float(robot['tcp_speed'][0])) <= SETDOWN_SPEED_MM_S)
            if near:
                if self.setdown_last is None or stamp - self.setdown_last > SETDOWN_GAP_S:
                    self.setdown_since = stamp
                self.setdown_last = stamp
            else:
                self.setdown_since = self.setdown_last = None
            state['setdown_age_s'] = None if self.setdown_since is None else stamp - self.setdown_since
        return {'state': state, 'camera_monotonic': stamp, 'feedback_monotonic': feedback_stamp,
                'contact_monotonic': self.contact_time, 'flange_rise_mm': height,
                'rim_rise_mm': rim_rise, 'pickup_reference_error_mm': flange_error.tolist(),
                'hover_start_monotonic': self.hover_start}
