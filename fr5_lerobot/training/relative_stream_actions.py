"""FR5 action-chunk adapter for the LeRobot 0.6.1 pi0.5 custom pipeline.

The model uses six pose offsets from one measured observation per chunk.
The pose is either joints or TCP coordinates in the cup frame. Cup-frame
offsets can also be expressed per 6 Hz tick to give near and far targets
comparable scale. The seventh component stays an absolute gripper target.
Do not combine this adapter with an enabled LeRobot relative/absolute pair.

The functions accept torch tensors and never mutate their inputs.  Numeric
values cannot identify their own representation: callers must pass the
representation carried by their checkpoint/batch metadata explicitly.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


ABSOLUTE = 'absolute-joint-position-v1'
RELATIVE = 'joint-delta-from-anchor-v1'
CUP_ABSOLUTE = 'cup-relative-tcp-v1'
CUP_RELATIVE = 'cup-relative-tcp-delta-v1'
CUP_TICK_RELATIVE = 'cup-relative-tcp-tick-delta-v1'
STATE_DIM, ACTION_DIM, JOINT_DIM, CHUNK = 8, 7, 6, 50


def _require_tensors(actions, state):
    if not isinstance(actions, torch.Tensor) or not isinstance(state, torch.Tensor):
        raise TypeError('actions and state must be torch tensors')
    if not actions.is_floating_point() or not state.is_floating_point():
        raise TypeError('actions and state must be floating point')
    if state.ndim not in (1, 2) or state.shape[-1] != STATE_DIM:
        raise ValueError('state must have shape [8] or [B,8]')
    if actions.ndim not in (1, 2, 3) or actions.shape[-1] != ACTION_DIM:
        raise ValueError('actions must end in 7 dimensions')
    if state.ndim == 1:
        if actions.ndim not in (1, 2):
            raise ValueError('an unbatched state requires [7] or [T,7] actions')
    elif actions.ndim not in (2, 3) or actions.shape[0] != state.shape[0]:
        raise ValueError('a batched state requires matching [B,7] or [B,T,7] actions')
    if actions.numel() == 0 or state.numel() == 0:
        raise ValueError('empty action or state')
    if not bool(torch.isfinite(actions).all()) or not bool(torch.isfinite(state).all()):
        raise ValueError('actions and state must be finite, including padding')


def _convert(actions, state, *, input_representation, expected, sign, per_tick=False):
    if input_representation not in expected:
        raise ValueError(f'Expected an action representation in {expected!r}, got {input_representation!r}')
    _require_tensors(actions, state)
    anchor = state[..., :JOINT_DIM].to(device=actions.device, dtype=actions.dtype)
    is_chunk = actions.ndim == state.ndim + 1
    if is_chunk:
        anchor = anchor.unsqueeze(-2)
    result = actions.clone()
    pose = result[..., :JOINT_DIM]
    if per_tick and is_chunk:
        horizons = torch.arange(1, actions.shape[-2]+1, device=actions.device, dtype=actions.dtype).unsqueeze(-1)
        pose[:] = (pose - anchor) / horizons if sign < 0 else pose * horizons + anchor
    else:
        pose[:] = pose + sign * anchor
    if not bool(torch.isfinite(result).all()):
        raise ValueError('action conversion overflowed')
    return result


def to_relative_actions(actions, state, *, input_representation, per_tick=False):
    """Convert [B,T,7] absolute targets with the *same* [B,8] anchor.

    Unbatched [T,7]/[8] and single-step [B,7]/[B,8] inputs also work.
    ``input_representation`` selects absolute joints or cup-frame TCP.
    For cup-frame TCP, per_tick divides each pose offset by its one-based
    horizon in the chunk. Action 6 is never changed.
    """
    if per_tick and input_representation != CUP_ABSOLUTE:
        raise ValueError('Per-tick action offsets require a cup-frame TCP input')
    return _convert(actions, state, input_representation=input_representation,
                    expected=(ABSOLUTE, CUP_ABSOLUTE), sign=-1, per_tick=per_tick)


def to_absolute_actions(actions, state, *, input_representation):
    """Restore model outputs to wire targets using the original request state."""
    return _convert(actions, state, input_representation=input_representation,
                    expected=(RELATIVE, CUP_RELATIVE, CUP_TICK_RELATIVE), sign=1,
                    per_tick=input_representation == CUP_TICK_RELATIVE)


@dataclass(frozen=True)
class TaggedActions:
    """Optional representation-carrying wrapper for avoiding repeat conversion."""
    values: torch.Tensor
    representation: str

    def relative(self, state):
        return TaggedActions(to_relative_actions(self.values, state,
                                                 input_representation=self.representation),
                             CUP_RELATIVE if self.representation == CUP_ABSOLUTE else RELATIVE)

    def absolute(self, state):
        return TaggedActions(to_absolute_actions(self.values, state,
                                                 input_representation=self.representation),
                             CUP_ABSOLUTE if self.representation in (CUP_RELATIVE, CUP_TICK_RELATIVE) else ABSOLUTE)


@dataclass(frozen=True)
class RequestAnchor:
    """A copy of one stdio request's raw state, never shared across requests."""
    request_id: str
    _state: torch.Tensor

    @classmethod
    def capture(cls, request_id, state):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('request_id must be a non-empty string')
        if not isinstance(state, torch.Tensor) or state.ndim != 1 or state.shape != (STATE_DIM,):
            raise ValueError('request state must be a torch tensor of shape [8]')
        if not state.is_floating_point() or not bool(torch.isfinite(state).all()):
            raise ValueError('request state must contain finite floating point values')
        return cls(request_id=request_id, _state=state.detach().clone())

    def restore(self, actions, *, request_id, input_representation):
        if request_id != self.request_id:
            raise ValueError('prediction request_id does not match its state anchor')
        return to_absolute_actions(actions, self._state, input_representation=input_representation)


def assert_manual_pipeline(config, preprocessor=None, postprocessor=None):
    """Reject a config/pipeline that would independently do the conversion.

    ``config`` can be a PI05Config or its JSON dict.  Pipeline arguments can be
    live processors with ``.steps`` or their JSON dict with a ``steps`` list.
    Call this at startup and additionally bind the checkpoint representation in
    the caller's manifest.  The helper cannot infer a tensor's semantics.
    """
    setting = (config.get('use_relative_actions') if isinstance(config, Mapping)
               else getattr(config, 'use_relative_actions', None))
    if setting is not False:
        raise ValueError('manual relative adapter requires use_relative_actions=False')
    for pipeline in (preprocessor, postprocessor):
        if pipeline is None:
            continue
        steps = pipeline.get('steps') if isinstance(pipeline, Mapping) else getattr(pipeline, 'steps', None)
        if steps is None:
            raise ValueError('cannot inspect processor steps')
        for step in steps:
            if isinstance(step, Mapping):
                name = step.get('registry_name', step.get('name', step.get('type', '')))
                enabled = step.get('config', {}).get('enabled', step.get('enabled', True))
            else:
                name = type(step).__name__
                enabled = getattr(step, 'enabled', True)
            if name in ('relative_actions_processor', 'absolute_actions_processor',
                        'RelativeActionsProcessorStep', 'AbsoluteActionsProcessorStep') and enabled:
                raise ValueError('enabled official relative/absolute step would duplicate manual conversion')


def metadata():
    """JSON-compatible declaration suitable for a new checkpoint manifest."""
    return {
        'schema': 'fr5-relative-stream-actions-v1',
        'canonical_dataset_and_wire': 'fr5-joint-position-stream-v1',
        'dataset_action_representation': ABSOLUTE,
        'model_action_representation': RELATIVE,
        'wire_action_representation': ABSOLUTE,
        'state': 'absolute-measured-8d',
        'joint_dimensions': list(range(JOINT_DIM)),
        'gripper_dimension': 6,
        'gripper_representation': 'absolute-fraction',
        'anchor': 'same-request-raw-observation-state-first-six-joints-radians',
        'all_chunk_steps_use_same_anchor': True,
        'config_use_relative_actions': False,
        'manual_adapter_requires_use_relative_actions': False,
        'manual_adapter_requires_official_relative_and_absolute_steps_disabled': True,
        'action_statistics': 'training-anchors-times-valid-chunk-targets-only',
        'state_statistics': 'absolute-state-training-sources',
        'reference_implementation': 'LeRobot 0.6.1 relative_action_processor mask=[true]*6+[false]',
    }


def _stats_from_values(values, *, representation=RELATIVE):
    if values.ndim != 2 or values.shape[1] != ACTION_DIM or len(values) == 0:
        raise ValueError('no valid action targets')
    if not np.isfinite(values).all():
        raise ValueError('non-finite action target')
    raw_low, raw_high = np.quantile(values, [.01, .99], axis=0)
    low, high = raw_low.copy(), raw_high.copy()
    fallbacks = []
    for i in range(ACTION_DIM):
        # TCP attitude is held fixed; milliradian-scale telemetry variation
        # should not carry as much weight as useful translation.
        near_constant = .0001 if representation == CUP_TICK_RELATIVE and i < 3 else .001
        if high[i] - low[i] < near_constant:
            observed_low, observed_high = float(values[:, i].min()), float(values[:, i].max())
            if i == JOINT_DIM:
                low[i], high[i] = 0., 1.
                reason = 'near_constant_gripper_physical_range'
            elif observed_high - observed_low >= near_constant and not (
                    representation == CUP_TICK_RELATIVE and 3 <= i < 6):
                low[i], high[i] = observed_low, observed_high
                reason = 'near_constant_quantiles_observed_train_range'
            else:
                center = (observed_low + observed_high) / 2
                width = .001 if representation in (CUP_RELATIVE, CUP_TICK_RELATIVE) else 1.
                low[i], high[i] = center - width/2., center + width/2.
                reason = (('near_constant_tcp_one_millimetre_scale' if i < 3 else
                           'near_constant_tcp_one_milliradian_scale') if width == .001
                          else 'near_constant_joint_one_radian_scale')
            fallbacks.append({'index': i, 'reason': reason})
    stats = {'action': {'q01': torch.tensor(low, dtype=torch.float32),
                        'q99': torch.tensor(high, dtype=torch.float32)}}
    report = {'valid_action_count': int(len(values)),
              'raw_q01': raw_low.tolist(), 'raw_q99': raw_high.tolist(),
              'mean': values.mean(axis=0).tolist(), 'std': values.std(axis=0).tolist(),
              'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist(),
              'fallbacks': fallbacks}
    return stats, report


def stats_from_padded_chunks(actions, states, valid_mask, *, splits):
    """Stats for actual [B,T,7] chunks, explicitly excluding invalid padding.

    Every supplied anchor must be marked ``train``.  Repeated rows are counted
    repeatedly, so callers may provide actual sampler-drawn anchors.
    """
    _require_tensors(actions, states)
    if actions.ndim != 3 or states.ndim != 2:
        raise ValueError('statistics require [B,T,7] actions and [B,8] states')
    if (not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool
            or tuple(valid_mask.shape) != tuple(actions.shape[:2])):
        raise ValueError('valid_mask must be bool [B,T], True for a real target')
    if len(splits) != len(states) or any(split != 'train' for split in splits):
        raise ValueError('all statistics anchors must belong to the training split')
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError('each anchor must have at least one real target')
    values = to_relative_actions(actions, states, input_representation=ABSOLUTE)
    selected = values[valid_mask.to(values.device)].detach().cpu().double().numpy()
    stats, report = _stats_from_values(selected)
    report.update(anchor_count=len(states), padding_count=int(valid_mask.numel()-valid_mask.sum().item()))
    return stats, report


def compute_relative_action_stats(rows, train_episodes, *, chunk_size=CHUNK, anchor_indices=None,
                                  representation=RELATIVE, camera_aligned=False):
    """Compute quantiles over every valid target of the selected training chunks.

    ``train_episodes`` maps episode IDs to their row indices, as returned by
    train_stream_pi05.split_sources.  By default every eligible anchor appears
    once.  ``anchor_indices`` can include duplicates to match a particular
    training sampler.  No target is drawn across an episode or from padding.
    """
    if representation not in (RELATIVE, CUP_RELATIVE, CUP_TICK_RELATIVE):
        raise ValueError(f'Expected a delta representation, got {representation!r}')
    state_key, action_key = (('_cup_relative_state', '_cup_relative_action')
                             if representation in (CUP_RELATIVE, CUP_TICK_RELATIVE) else
                             ('observation.state', 'action'))
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if not isinstance(train_episodes, Mapping) or not train_episodes:
        raise ValueError('training episodes are required')
    offsets, sources, episode_ids = {}, set(), []
    for ep, indices in train_episodes.items():
        indices = list(indices)
        if not indices:
            raise ValueError('empty training episode')
        if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
               or not 0 <= i < len(rows) for i in indices):
            raise ValueError('invalid training episode index')
        if any(b != a + 1 for a, b in zip(indices, indices[1:])):
            raise ValueError('episode indices must be contiguous and ordered')
        source = None
        for position, index in enumerate(indices):
            row = rows[index]
            row_source = row.get('_physical_source')
            if row.get('_split') != 'train' or row_source is None:
                raise ValueError('statistics may only use labeled training sources')
            row_source = str(row_source)
            if source is None:
                source = row_source
            if source != row_source or index in offsets:
                raise ValueError('episode crosses sources or repeats a row')
            if 'episode_index' in row and int(row['episode_index']) != int(ep):
                raise ValueError('row episode does not match episode mapping')
            if 'frame_index' in row and int(row['frame_index']) != position:
                raise ValueError('episode frame indices are not contiguous')
            offsets[int(index)] = (indices, position)
        sources.add(source)
        episode_ids.append(int(ep))
    for row in rows:
        source = row.get('_physical_source')
        if source is not None and str(source) in sources and row.get('_split') != 'train':
            raise ValueError('physical source is present in a non-training split')
    anchors = list(offsets) if anchor_indices is None else list(anchor_indices)
    if not anchors or any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
                          or int(i) not in offsets for i in anchors):
        raise ValueError('all selected anchors must belong to a training episode')
    values, padding = [], 0
    for index in anchors:
        indices, position = offsets[int(index)]
        # Training constructs float32 tensors before subtracting the anchor;
        # reproduce that arithmetic before computing high-precision quantiles.
        state = np.asarray(rows[int(index)][state_key], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError('invalid absolute anchor state')
        targets = (np.array(rows[int(index)]['_camera_aligned_actions'][:chunk_size], dtype=np.float32) if camera_aligned else
                   np.asarray([rows[i][action_key] for i in indices[position:position + chunk_size]], dtype=np.float32))
        if targets.ndim != 2 or targets.shape[1] != ACTION_DIM or not len(targets) or not np.isfinite(targets).all():
            raise ValueError('invalid absolute action target')
        with np.errstate(over='ignore', invalid='ignore'):
            targets[:, :JOINT_DIM] -= state[:JOINT_DIM]
            if representation == CUP_TICK_RELATIVE:
                targets[:, :JOINT_DIM] /= np.arange(1, len(targets)+1, dtype=np.float32)[:, None]
        if not np.isfinite(targets).all():
            raise ValueError('relative action conversion overflowed')
        values.append(targets.astype(np.float64))
        padding += chunk_size-len(targets)
    stats, report = _stats_from_values(np.concatenate(values, axis=0), representation=representation)
    report.update({
        'schema': 'fr5-relative-action-chunk-stats-v1',
        'representation': representation,
        'anchor_selection': 'all_eligible_training_anchors_once' if anchor_indices is None else 'explicit_anchor_sequence',
        'anchor_count': len(anchors), 'unique_anchor_count': len(set(int(i) for i in anchors)),
        'chunk_size': chunk_size, 'padding_count': padding,
        'train_episode_indices': episode_ids, 'train_physical_sources': sorted(sources),
        'heldout_included': False,
    })
    return stats, report
