"""Pure motion-profile definitions; usable without recorder dependencies."""
SPEEDS = {'empty_free':5., 'held_free':3., 'near_contact':1.}
LEGACY_PROFILE = 'legacy'
CONTINUOUS_PROFILE = 'continuous-supervised-v1'
PROFILE_SPEEDS = {
    LEGACY_PROFILE: SPEEDS,
    CONTINUOUS_PROFILE: {'empty_free':10., 'held_free':3., 'near_contact':1.},
}
FREE_PHASES = {'approach_clear', 'approach_near', 'retreat_clear', 'retreat_high'}
PROBE_PHASES = {'approach_contact', 'lift_probe', 'retreat_probe'}


def validate_profile(profile):
    if not isinstance(profile, str) or profile not in PROFILE_SPEEDS:
        raise ValueError('Unknown supervised motion profile')
    return profile


def record_profile(record):
    # Plans, geometry and scopes prepared before profiles existed are legacy.
    return validate_profile(record.get('profile', LEGACY_PROFILE))


def matching_profiles(*records):
    profiles = [record_profile(record) for record in records]
    if len(set(profiles)) != 1:
        raise ValueError('Motion profile mismatch between scope, geometry and action plan')
    return profiles[0]


def motion_limits(profile, phase, source_pose, target_pose, support_z):
    """Return planned max distance and speed category for one route section."""
    validate_profile(profile)
    target_above = float(target_pose[2]) - float(support_z)
    if profile == LEGACY_PROFILE:
        if phase in PROBE_PHASES:
            step = 5.
        elif phase == 'return':
            step = 10. if target_above > 1.9 else 1.
        else:
            step = 15.
        if phase in FREE_PHASES:
            category = 'empty_free'
        elif phase == 'lift_full' or (phase == 'return' and target_above > 1.9):
            category = 'held_free'
        else:
            category = 'near_contact'
        return step, category

    if phase in FREE_PHASES:
        return 80., 'empty_free'
    if phase == 'lift_full':
        return 40., 'held_free'
    if phase in PROBE_PHASES:
        return 10., 'near_contact'
    if phase == 'return':
        # The whole section must be above the final 2 mm to use held speed.
        # A section crossing the boundary must obey the slower, shorter limit.
        source_above = float(source_pose[2]) - float(support_z)
        if min(source_above, target_above) >= 2. - 1e-8:
            return 40., 'held_free'
        return 1., 'near_contact'
    raise ValueError('No arm motion is defined for this phase')


