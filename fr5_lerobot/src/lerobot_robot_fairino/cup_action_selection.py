"""Bounded selection of complete π0.5 draws at measured cup boundaries."""

import math

from .cup_relative_runtime import check_cup_gripper_change


def supported_open_tick(request, normal, actions, ticks=3):
    timing = request['cup_timing']
    if (request['cup_tracking']['native_contact'] != 1 or timing['setdown_age_s'] is None
            or timing['setdown_age_s'] < .30 or timing['support_gap_mm'] is None
            or abs(timing['support_gap_mm']) > 1.5):
        return None
    origin = previous = request['cup_relative_state'][:3]
    opening = round(request['cup_relative_state'][6] * 100)
    first_full_open = None
    for tick, action in enumerate(actions[:ticks], 1):
        target_opening = min(100, max(0, round(action[6] * 100)))
        target = action[:3]
        if target_opening - opening > 1 and opening < 98:
            gap = timing['support_gap_mm'] + 1000. * sum((x - y) * n for x, y, n in zip(target, origin, normal))
            previous_gap = timing['support_gap_mm'] + 1000. * sum(
                (x - y) * n for x, y, n in zip(previous, origin, normal))
            if max(abs(gap), abs(previous_gap)) > 1.5 or math.dist(target, previous) * 6000. > 4.:
                return None
        if first_full_open is not None and target_opening < 98:
            return None
        if target_opening >= 98 and first_full_open is None:
            first_full_open = tick
        if abs(target_opening - opening) > 1:
            opening = target_opening
        previous = target
    return first_full_open


def opening_while_held(request, normal, actions):
    timing = request['cup_timing']
    gap = timing['support_gap_mm']
    origin = request['cup_relative_state'][:3]
    opening = round(request['cup_relative_state'][6] * 100)
    for action in actions[:3]:
        target_opening = min(100, max(0, round(action[6] * 100)))
        if target_opening - opening > 1 and opening < 98:
            if gap is None or abs(gap + 1000. * sum((x - y) * n for x, y, n in zip(action, origin, normal))) > 1.5:
                return True
        if abs(target_opening - opening) > 1:
            opening = target_opening
    return False


def close_at_pickup(request):
    state = request['cup_relative_state']
    prior = request['observation_history'][0]
    pickup = (159.7, 71.3, 95.8)
    rim = request['cup_tracking']['rim_minus_jaw_mm']
    return (request['cup_timing']['phase'] == 'at_pickup' and request['cup_tracking']['native_contact'] == 0
            and state[7] == 1. and state[6] >= .98 and rim is not None
            and math.dist(rim, (-13.5, -3.4, 13.9)) <= 15.
            and math.dist([x * 1000. for x in state[:3]], pickup) <= 2.
            and prior is not None and prior['age_s'] <= .7 and prior['state'][6] >= .98 and prior['state'][7] == 1.
            and math.dist(state[:3], prior['state'][:3]) <= .0015)


def choose_proposal(model, request, normal, *, check_boundaries=False):
    """Select a complete draw using only current observations and model actions."""
    ready = request['cup_timing']['setdown_age_s']
    holding = request['cup_tracking']['native_contact'] == 1
    if holding and ready is not None and ready >= .30:
        rank = lambda actions: supported_open_tick(request, normal, actions)
        desired = 1
    elif check_boundaries and holding and request['cup_timing']['phase'] in ('returning', 'at_table_height'):
        rank = lambda actions: None if opening_while_held(request, normal, actions) else 1
        desired = 1
    elif check_boundaries and close_at_pickup(request):
        pickup = (159.7, 71.3, 95.8)
        def rank(actions):
            return next((tick for tick, action in enumerate(actions[:2], 1)
                         if round(action[6] * 100) <= 77
                         and math.dist([x * 1000. for x in action[:3]], pickup) <= 3.
                         and all(round(a[6] * 100) <= 88 for a in actions[tick:3])), None)
        desired = 2
    elif (check_boundaries and not holding and request['cup_relative_state'][7] == 1.
          and request['cup_timing']['phase'] in ('approaching', 'at_pickup')):
        def rank(actions):
            opening = min(100, max(0, round(request['cup_relative_state'][6] * 100)))
            for action in actions[:3]:
                target = min(100, max(0, round(action[6] * 100)))
                try:
                    check_cup_gripper_change(action, opening, target)
                except RuntimeError:
                    return None
                if abs(target - opening) > 1:
                    opening = target
            return 1
        desired = 1
    else:
        return model.predict(request), 1, False

    proposals = []
    candidates = []
    draw_results = []
    for i in range(3):
        proposal = model.predict(request)
        proposals.append(proposal)
        score = rank(proposal['actions'])
        draw_results.append({'actions': proposal['actions'],
                             'inference_seconds': proposal['inference_seconds'],
                             'matched': score is not None,
                             'sampling': proposal.get('sampling'),
                             'runtime_source': proposal.get('runtime_source')})
        if score is not None:
            candidates.append((score, i))
            if score <= desired:
                break
    chosen = min(candidates)[1] if candidates else 0
    reply = dict(proposals[chosen], inference_seconds=sum(r['inference_seconds'] for r in proposals))
    reply['cup_proposal_selection'] = {
        'draws': len(proposals), 'selected_index': chosen, 'matched': bool(candidates),
        'proposals': draw_results}
    return reply, len(proposals), chosen > 0
