"""Train FR5 pi0.5 on continuous 6 Hz seven-dimensional position targets.

Joint targets are derived from measured feedback; gripper targets are latched
from recorded commands or the explicitly recorded non-RPC initial hold.
Train/validation are split by physical source; no robot I/O occurs here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import copy
from functools import lru_cache
import hashlib
import inspect
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

import stream_checkpoint as checkpoints
import stream_prompt_integrity as prompt_integrity
import stream_conditioned_sampling as conditioned_sampling
import stream_observation_history as observation_history
import train_pi05_fr5 as legacy

FPS, CHUNK, STATE_DIM, ACTION_DIM = 6, 50, 8, 7
IMAGE_KEY = 'observation.images.front'
LORA_TARGETS = checkpoints.BASE_LORA_TARGETS
VISION_LORA_TARGETS = checkpoints.VISION_LORA_TARGETS
VISION_EXPERT_LORA_TARGETS = checkpoints.VISION_EXPERT_LORA_TARGETS
PROJECTION_PREFIXES = ('model.action_in_proj.', 'model.action_out_proj.')
LORA_ADAPTATION = checkpoints.LORA_ADAPTATION
FULL_EXPERT_ADAPTATION = checkpoints.FULL_EXPERT_ADAPTATION
VISION_EXPERT_ADAPTATION = checkpoints.VISION_EXPERT_ADAPTATION
FULL_EXPERT_MODES = checkpoints.FULL_EXPERT_MODES
ADAPTATION_MODES = checkpoints.ADAPTATION_MODES
ABSOLUTE_REPRESENTATION = checkpoints.ABSOLUTE_REPRESENTATION
RELATIVE_REPRESENTATION = checkpoints.RELATIVE_REPRESENTATION
FULL_IMAGE_PROFILE = checkpoints.FULL_IMAGE_PROFILE
DEFAULT_SAMPLING_PROFILE = 'balanced-v2'
APPROACH_SAMPLING_PROFILE = 'approach-open-v1'
EARLY_SWITCH_SAMPLING_PROFILE = 'early-switch-context-v1'
CUP_TIMING_SAMPLING_PROFILE = 'cup-timing-contact-return-v1'
CUP_CLOCK_SAMPLING_PROFILE = 'cup-clock-hold-return-v1'
CUP_EVENT_SAMPLING_PROFILE = 'cup-camera-clock-events-v1'
CUP_BOUNDARY_SAMPLING_PROFILE = 'cup-camera-clock-boundaries-v1'
CUP_SUPPORT_SAMPLING_PROFILE = 'cup-camera-clock-support-v1'
CUP_COUPLED_SAMPLING_PROFILE = 'cup-camera-clock-coupled-support-v1'
UNIFORM_SAMPLING_PROFILE = 'uniform-all-frames-v1'
UNIFORM_SWITCH_PROFILE = 'uniform-switch-v1'
UNIFORM_SWITCH_HOLD_PROFILE = 'uniform-switch-hold-v1'
SAMPLING_PROFILES = (DEFAULT_SAMPLING_PROFILE, APPROACH_SAMPLING_PROFILE, EARLY_SWITCH_SAMPLING_PROFILE,
                     CUP_TIMING_SAMPLING_PROFILE, CUP_CLOCK_SAMPLING_PROFILE, CUP_EVENT_SAMPLING_PROFILE,
                     CUP_BOUNDARY_SAMPLING_PROFILE, CUP_SUPPORT_SAMPLING_PROFILE,
                     CUP_COUPLED_SAMPLING_PROFILE, UNIFORM_SAMPLING_PROFILE,
                     UNIFORM_SWITCH_PROFILE, UNIFORM_SWITCH_HOLD_PROFILE)


def load_contract():
    for path in (Path(__file__).with_name('stream_contract.py'),
                 Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'):
        if path.is_file():
            spec = importlib.util.spec_from_file_location('fr5_train_stream_contract', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, path
    raise FileNotFoundError('FR5 stream contract not found')


def load_record_helpers():
    """Load source-relative helpers even when the CLI is run directly."""
    script = Path(__file__).resolve()
    source = script.parents[1] / 'src'
    modules = []
    for name, relative, bundled in (('fr5_train_camera_timing', 'lerobot_camera_gemini/timing.py', 'camera_timing.py'),
                                    ('fr5_train_gripper_feedback', 'lerobot_robot_fairino/feedback.py', 'gripper_feedback.py')):
        candidates = (script.parent / 'record_helpers' / bundled, source / relative)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Recorded-feedback helper missing: {relative}')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


@lru_cache(maxsize=1)
def load_relative_helper():
    """The sealed converter is adjacent to this source or checkpoint."""
    path = Path(__file__).resolve().with_name('relative_stream_actions.py')
    if not path.is_file():
        raise FileNotFoundError(f'Relative action converter missing: {path}')
    spec = importlib.util.spec_from_file_location('fr5_relative_action_helper', path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_image_profile_helper():
    """Load the same profile implementation from the source or sealed bundle."""
    path = Path(__file__).resolve().with_name('stream_image_profile.py')
    if not path.is_file():
        raise FileNotFoundError(f'Image-profile helper missing: {path}')
    spec = importlib.util.spec_from_file_location('fr5_stream_image_profile_helper', path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def apply_image_profile(batch, name, *, initial_cup_uv=None):
    helper = load_image_profile_helper()
    if helper.image_profile_metadata(name) != checkpoints.image_profile_metadata(name):
        raise ValueError('Image-profile helper disagrees with sealed metadata')
    if name == checkpoints.CUP_CENTERED_IMAGE_PROFILE:
        return helper.apply_image_profile(batch, name, initial_cup_uv=initial_cup_uv)
    return helper.apply_image_profile(batch, name)


def action_representation_metadata(mode):
    if mode in checkpoints.CUP_REPRESENTATIONS:
        return checkpoints.cup_action_metadata(mode)
    if mode not in (ABSOLUTE_REPRESENTATION, RELATIVE_REPRESENTATION):
        raise ValueError('Unknown action representation')
    return {'schema': 'fr5-stream-action-representation-v1', 'mode': mode,
            'wire_mode': ABSOLUTE_REPRESENTATION, 'joint_dimensions': 6,
            'gripper_representation': 'absolute',
            'anchor': 'raw_request_state' if mode == RELATIVE_REPRESENTATION else None,
            'official_relative_processor_enabled': False}


def _scalar(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def validate_rows(rows):
    """Require a single, complete, uniformly sampled sequence per episode."""
    import numpy as np
    if not rows:
        raise ValueError('Dataset is empty')
    episodes = defaultdict(list)
    for index, row in enumerate(rows):
        if int(_scalar(row['index'])) != index:
            raise ValueError('Dataset index is not contiguous')
        state, action = np.asarray(row['observation.state']), np.asarray(row['action'])
        if (state.shape != (STATE_DIM,) or action.shape != (ACTION_DIM,)
                or not np.isfinite(state).all() or not np.isfinite(action).all()
                or not 0 <= state[6] <= 1 or state[7] not in (0, 1) or not 0 <= action[6] <= 1):
            raise ValueError(f'Invalid state/action at row {index}')
        ep = int(_scalar(row['episode_index']))
        source = row.get('_physical_source', row.get('source.trial', row.get('source_trial')))
        if source is None:
            raise ValueError('Each row requires its physical source.trial')
        row['_physical_source'] = str(_scalar(source))
        episodes[ep].append(index)
    for ep, indices in episodes.items():
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f'Episode {ep} is not contiguous')
        if len({rows[i]['_physical_source'] for i in indices}) != 1:
            raise ValueError(f'Episode {ep} crosses physical sources')
        for n, i in enumerate(indices):
            row = rows[i]
            if 'frame_index' in row and int(_scalar(row['frame_index'])) != n:
                raise ValueError(f'Episode {ep} has missing frames')
            if n and 'timestamp' in row:
                dt = float(_scalar(row['timestamp'])) - float(_scalar(rows[indices[n-1]]['timestamp']))
                if abs(dt - 1/FPS) > .002:
                    raise ValueError(f'Episode {ep} is not sampled at 6 Hz')
    return dict(episodes)


def split_sources(rows, episodes):
    ordered = list(dict.fromkeys(row['_physical_source'] for row in rows))
    if len(ordered) < 2:
        raise ValueError('At least two physical sources are required for source holdout')
    splits = {}
    for row in rows:
        source, split = row['_physical_source'], row.get('_split')
        if split is None:
            split = 'validation' if source == ordered[-1] else 'train'
        if split not in ('train', 'validation') or (source in splits and splits[source] != split):
            raise ValueError('A physical source must belong to exactly one split')
        splits[source] = split
    heldout = [source for source in ordered if splits[source] == 'validation']
    train_eps = {ep: indices for ep, indices in episodes.items() if splits[rows[indices[0]]['_physical_source']] == 'train'}
    val_eps = {ep: indices for ep, indices in episodes.items() if splits[rows[indices[0]]['_physical_source']] == 'validation'}
    if not train_eps or not val_eps:
        raise ValueError('Empty training or holdout split')
    return train_eps, val_eps, heldout


def chunk_for_anchor(rows, indices, offset, *, action_key='action'):
    """Retain each real suffix target; pads have no loss or attention access."""
    targets = (rows[indices[offset]][action_key] if action_key == '_camera_aligned_actions' else
               [rows[i][action_key] for i in indices[offset:offset + CHUNK]])
    if len(targets) == 0:
        raise ValueError('Anchor outside its episode')
    count = len(targets)
    targets = [list(x) for x in targets] + [list(targets[-1]) for _ in range(CHUNK-count)]
    return targets, [True]*count + [False]*(CHUNK-count)


def calculate_stats(rows, train_episodes, exported=None, *, cup_relative=False, cup_delta=False):
    import numpy as np
    import torch
    indices = [i for ep in train_episodes.values() for i in ep]
    result, details = {}, []
    for feature, dim in (('observation.state', STATE_DIM), ('action', ACTION_DIM)):
        key = ('_cup_relative_state' if feature == 'observation.state' else '_cup_relative_action') if cup_relative else feature
        values = np.asarray([rows[i][key] for i in indices], dtype=np.float64)
        low, high = np.quantile(values, [.01, .99], axis=0)
        if exported is not None:
            low = np.asarray(exported[feature]['q01'], dtype=np.float64).copy()
            high = np.asarray(exported[feature]['q99'], dtype=np.float64).copy()
        if cup_delta and feature == 'observation.state':
            # π0.5 bins state into 256 tokens. Keep a margin for approach
            # poses beyond the demonstrations' measured position range.
            position_margin_m = .080
            low[:3] = values[:, :3].min(axis=0) - position_margin_m
            high[:3] = values[:, :3].max(axis=0) + position_margin_m
            details.extend({'feature': feature, 'index': i,
                            'reason': 'observed_train_position_range_with_80mm_margin'} for i in range(3))
        for i in range(dim):
            if high[i] - low[i] < 1e-3:
                observed_low, observed_high = float(values[:, i].min()), float(values[:, i].max())
                ratio = (feature == 'observation.state' and i in (6, 7)) or (feature == 'action' and i == 6)
                if ratio:
                    low[i], high[i] = 0., 1.
                    reason = 'near_constant_fraction_physical_range'
                elif observed_high - observed_low >= 1e-3:
                    low[i], high[i] = observed_low, observed_high
                    reason = 'near_constant_quantiles_observed_train_range'
                else:
                    center = (observed_low + observed_high) / 2
                    low[i], high[i] = center - .5, center + .5
                    reason = 'near_constant_unit_scale' if cup_relative else 'near_constant_joint_one_radian_scale'
                details.append({'feature': feature, 'index': i, 'reason': reason})
        result[feature] = {'q01': torch.tensor(low, dtype=torch.float32),
                           'q99': torch.tensor(high, dtype=torch.float32)}
    return result, details


def resolve_cup_roll_state_min_span(requested, *, parent_report=None, resume=False, cup_relative=False):
    parent_span = parent_report.get('cup_roll_state_min_span_rad') if parent_report else None
    span = parent_span if requested is None else requested
    if span is not None and (isinstance(span, bool) or not isinstance(span, (int, float))
                             or not math.isfinite(span) or span <= 0.):
        raise ValueError('Cup roll state minimum span must be positive and finite')
    if span is not None and not cup_relative:
        raise ValueError('Cup roll state minimum span requires cup-relative state')
    if resume and requested is not None and requested != parent_span:
        raise ValueError('Resume cannot change cup roll state normalization')
    return span


def cup_roll_state_bounds(training_min, training_max, minimum_span, existing_bounds):
    """Center the state roll scale on recorded training-source extrema."""
    import numpy as np
    if not (math.isfinite(training_min) and math.isfinite(training_max)
            and training_min <= training_max and math.isfinite(minimum_span) and minimum_span > 0.):
        raise ValueError('Invalid training roll range or minimum span')
    low, high = (float(x) for x in existing_bounds)
    width = max(minimum_span, training_max - training_min, high - low)
    center = (training_min + training_max) / 2.
    low, high = np.float32(center - width / 2.), np.float32(center + width / 2.)
    if float(high) - float(low) < width:
        low = np.nextafter(low, np.float32(-np.inf))
        high = np.nextafter(high, np.float32(np.inf))
    return float(low), float(high)


def apply_cup_roll_state_min_span(stats, rows, train_episodes, minimum_span):
    """Change only roll normalization, using measured training observations."""
    indices = [i for indices in train_episodes.values() for i in indices]
    if not indices or any(rows[i].get('_split') != 'train' for i in indices):
        raise ValueError('Cup roll bounds require training-source observations')
    if {int(rows[i]['_physical_source']) for i in indices} != set(range(7)):
        raise ValueError('Cup roll bounds require original training sources 0–6')
    values = [float(rows[i]['_cup_relative_state'][3]) for i in indices]
    training_min, training_max = min(values), max(values)
    state = stats['observation.state']
    bounds = cup_roll_state_bounds(training_min, training_max, minimum_span,
                                  (state['q01'][3], state['q99'][3]))
    updated = dict(stats)
    updated['observation.state'] = {key: value.clone() for key, value in state.items()}
    updated['observation.state']['q01'][3] = bounds[0]
    updated['observation.state']['q99'][3] = bounds[1]
    return updated, {'training_min_rad': training_min, 'training_max_rad': training_max,
                     'minimum_span_rad': float(minimum_span)}


def configure_cup_roll_state_stats(stats, rows, train_episodes, requested, *, parent_report=None,
                                   resume=False, cup_relative=False, rebase_stats=False):
    span = resolve_cup_roll_state_min_span(requested, parent_report=parent_report,
                                          resume=resume, cup_relative=cup_relative)
    if parent_report is not None and rebase_stats and span is not None:
        raise ValueError('Cup roll span requires the parent action and state normalization')
    if requested is not None and not resume:
        updated, training_range = apply_cup_roll_state_min_span(stats, rows, train_episodes, span)
        return updated, span, training_range, True
    training_range = parent_report.get('cup_roll_state_training_range_rad') if parent_report else None
    return stats, span, training_range, False


def sampling_weights(rows, indices):
    """Mix uniform anchors with measured next-step movement, without phases."""
    import numpy as np
    movement = []
    for i in indices:
        state = np.asarray(rows[i]['observation.state'], dtype=np.float64)
        target = np.asarray(rows[i]['action'], dtype=np.float64)
        movement.append(float(np.linalg.norm(target[:6] - state[:6]) / math.sqrt(6)
                              + (abs(target[6] - state[6]) if state[7] == 1 else 0.)))
    scale = float(np.quantile(movement, .9))
    if scale < 1e-9:
        return [1.] * len(indices)
    return [1. + min(3., x / scale) for x in movement]


def transition_events(rows, episodes):
    """Find changes in real issued gripper targets, within each source clip."""
    events = {'close': [], 'open': []}
    for ep, indices in episodes.items():
        for position in range(1, len(indices)):
            previous, current = indices[position-1], indices[position]
            old, new = float(rows[previous]['action'][6]), float(rows[current]['action'][6])
            if abs(new-old) <= 1e-5:
                continue
            ref = rows[current].get('_gripper_target_ref')
            if (not isinstance(ref, dict) or ref.get('kind') != 'gripper'
                    or ref.get('executed') is not True or ref.get('source') != 'wire_command'
                    or abs(float(ref['target_degrees_percent'][6])/100. - new) > 1e-5):
                raise ValueError('Changed gripper target lacks matching issued-command provenance')
            kind = 'close' if new < old else 'open'
            events[kind].append({'episode': ep, 'position': position, 'index': current,
                                 'source': rows[current]['_physical_source'],
                                 'command_sequence': ref['sequence'], 'old': old, 'new': new})
    return events


class UniformFrameSampler:
    """Visit every training anchor once per independently shuffled epoch."""
    VERSION = UNIFORM_SAMPLING_PROFILE

    def __init__(self, rows, episodes, proof, seed):
        self.rows = rows
        self.anchors = [index for indices in episodes.values() for index in indices]
        if not self.anchors or len(set(self.anchors)) != len(self.anchors):
            raise ValueError('Training anchors must be nonempty and unique')
        self.rng = random.Random(seed)
        self.order = []
        self.cursor = 0
        self.epoch = 0

    def draw_balanced(self, count):
        if count < 1:
            raise ValueError('Sample count must be positive')
        result = []
        for _ in range(count):
            if self.cursor == len(self.order):
                self.order = self.anchors.copy()
                self.rng.shuffle(self.order)
                self.cursor = 0
                self.epoch += 1
            index = self.order[self.cursor]
            self.cursor += 1
            result.append((index, {'category': 'uniform', 'source': self.rows[index]['_physical_source'],
                                   'epoch': self.epoch, 'epoch_offset': self.cursor-1}))
        return result

    def getstate(self):
        return {'version': self.VERSION, 'rng': self.rng.getstate(), 'anchors': self.anchors,
                'order': self.order, 'cursor': self.cursor, 'epoch': self.epoch}

    def setstate(self, state):
        if (not isinstance(state, dict) or state.get('version') != self.VERSION
                or state.get('anchors') != self.anchors
                or (state.get('order') and sorted(state['order']) != sorted(self.anchors))
                or not 0 <= state.get('cursor', -1) <= len(state.get('order', []))):
            raise ValueError('Sampler state belongs to a different training dataset')
        self.rng.setstate(state['rng'])
        self.order = list(state['order'])
        self.cursor, self.epoch = int(state['cursor']), int(state['epoch'])

    def report(self):
        return {'version': self.VERSION, 'anchors_per_epoch': len(self.anchors),
                'epoch': self.epoch, 'epoch_samples': self.cursor,
                'completed_epochs': self.epoch - int(self.cursor < len(self.order))}


class UniformSwitchSampler:
    """Mix uniform frames with the recorded close and open transition anchors."""
    VERSION = UNIFORM_SWITCH_PROFILE

    def __init__(self, rows, episodes, proof, seed):
        self.uniform = UniformFrameSampler(rows, episodes, proof, seed)
        self.events = transition_events(rows, episodes)
        if not self.events['close'] or not self.events['open']:
            raise ValueError('Training sources need actual close and open target changes')
        self.rng = random.Random(seed)
        self.pending = {'close': [], 'open': []}
        self.counts = {kind: {str(event['index']): 0 for event in events}
                       for kind, events in self.events.items()}

    def draw_balanced(self, count):
        if count < 4 or count % 4:
            raise ValueError('Effective batch must be a positive multiple of four')
        result = []
        for _ in range(count // 4):
            group = self.uniform.draw_balanced(2)
            for kind in ('close', 'open'):
                queue = self.pending[kind]
                if not queue:
                    queue.extend(range(len(self.events[kind])))
                    self.rng.shuffle(queue)
                event = self.events[kind][queue.pop()]
                self.counts[kind][str(event['index'])] += 1
                group.append((event['index'], {'category': kind, 'event_index': event['index'],
                    'source': event['source'], 'offset_frames': 0,
                    'command_sequence': event['command_sequence']}))
            self.rng.shuffle(group)
            result.extend(group)
        return result

    def getstate(self):
        return {'version': self.VERSION, 'uniform': self.uniform.getstate(), 'rng': self.rng.getstate(),
                'event_indices': {kind: [event['index'] for event in events]
                                  for kind, events in self.events.items()},
                'pending': copy.deepcopy(self.pending), 'counts': copy.deepcopy(self.counts)}

    def setstate(self, state):
        expected = {kind: [event['index'] for event in events] for kind, events in self.events.items()}
        if not isinstance(state, dict) or state.get('version') != self.VERSION or state.get('event_indices') != expected:
            raise ValueError('Sampler state belongs to a different transition dataset')
        self.uniform.setstate(state['uniform'])
        self.rng.setstate(state['rng'])
        self.pending = copy.deepcopy(state['pending'])
        self.counts = copy.deepcopy(state['counts'])

    def report(self):
        return {'version': self.VERSION, 'group': ['uniform', 'uniform', 'close.at', 'open.at'],
                'uniform': self.uniform.report(), 'events': self.events, 'event_counts': self.counts}


class UniformSwitchHoldSampler(UniformSwitchSampler):
    """Mix uniform and switch anchors with recorded hold boundary frames."""
    VERSION = UNIFORM_SWITCH_HOLD_PROFILE

    def __init__(self, rows, episodes, proof, seed):
        super().__init__(rows, episodes, proof, seed)
        sources = {str(source['source_index']): source for source in proof['sources']}
        source_episodes = defaultdict(list)
        for episode, indices in episodes.items():
            source = rows[indices[0]]['_physical_source']
            if sources[source]['split'] != 'train':
                raise ValueError('Hold boundary sampling requires training sources only')
            source_episodes[source].append((episode, indices))
        self.hold_boundaries = []
        for source, clips in source_episodes.items():
            for hold_index, hold in enumerate(sources[source]['hold_records']):
                start, end = float(hold['start_monotonic']), float(hold['end_monotonic'])
                matches = []
                for episode, indices in clips:
                    inside = [i for i in indices if rows[i]['_physical_source'] == source
                              and start <= rows[i]['_grid_time'] <= end]
                    if inside:
                        matches.append((episode, inside))
                if len(matches) != 1:
                    raise ValueError('Each recorded hold must match frames in one training episode')
                episode, inside = matches[0]
                self.hold_boundaries.append({'source': source, 'episode': episode, 'hold_index': hold_index,
                    'start_monotonic': start, 'end_monotonic': end, 'start': inside[0], 'end': inside[-1]})
        if not self.hold_boundaries:
            raise ValueError('Training sources need recorded hold boundaries')
        self.pending_hold = {'start': [], 'end': []}
        self.hold_counts = {kind: [0] * len(self.hold_boundaries) for kind in self.pending_hold}
        self.next_hold_boundary = 'start'

    def draw_balanced(self, count):
        if count < 4 or count % 4:
            raise ValueError('Effective batch must be a positive multiple of four')
        result = []
        for _ in range(count // 4):
            group = self.uniform.draw_balanced(1)
            for kind in ('close', 'open'):
                queue = self.pending[kind]
                if not queue:
                    queue.extend(range(len(self.events[kind])))
                    self.rng.shuffle(queue)
                event = self.events[kind][queue.pop()]
                self.counts[kind][str(event['index'])] += 1
                group.append((event['index'], {'category': kind, 'event_index': event['index'],
                    'source': event['source'], 'offset_frames': 0,
                    'command_sequence': event['command_sequence']}))
            kind = self.next_hold_boundary
            queue = self.pending_hold[kind]
            if not queue:
                queue.extend(range(len(self.hold_boundaries)))
                self.rng.shuffle(queue)
            hold_index = queue.pop()
            hold = self.hold_boundaries[hold_index]
            self.hold_counts[kind][hold_index] += 1
            self.next_hold_boundary = 'end' if kind == 'start' else 'start'
            group.append((hold[kind], {'category': 'recorded_hold_boundary', 'boundary': kind,
                'source': hold['source'], 'episode': hold['episode'], 'hold_index': hold['hold_index'],
                'start_monotonic': hold['start_monotonic'], 'end_monotonic': hold['end_monotonic']}))
            self.rng.shuffle(group)
            result.extend(group)
        return result

    def getstate(self):
        state = super().getstate()
        state.update({'hold_boundaries': copy.deepcopy(self.hold_boundaries),
                      'pending_hold': copy.deepcopy(self.pending_hold),
                      'hold_counts': copy.deepcopy(self.hold_counts),
                      'next_hold_boundary': self.next_hold_boundary})
        return state

    def setstate(self, state):
        if not isinstance(state, dict) or state.get('hold_boundaries') != self.hold_boundaries:
            raise ValueError('Sampler state belongs to different recorded hold boundaries')
        super().setstate(state)
        self.pending_hold = copy.deepcopy(state['pending_hold'])
        self.hold_counts = copy.deepcopy(state['hold_counts'])
        self.next_hold_boundary = state['next_hold_boundary']

    def report(self):
        report = super().report()
        report.update({'group': ['uniform', 'close.at', 'open.at', 'recorded_hold_boundary'],
                       'hold_boundaries': self.hold_boundaries, 'hold_boundary_counts': self.hold_counts})
        return report


class TransitionSampler:
    """Balanced causal windows plus measured movement, recorded holds and all frames.

    Each four-example group contains close, open, motion and a full-trajectory
    example. The latter rotates between unrestricted and recorded-hold frames.
    Event queues cover each switch at and around its recorded target change.
    """
    VERSION = 'source-transition-motion-hold-balanced-batch-v2'
    SCHEDULE = ('close',)*5 + ('open',)*5 + ('motion',)*5 + ('hold',)*3 + ('all',)*2
    FOURTH_SCHEDULE = ('all', 'hold', 'all', 'hold', 'all')

    def __init__(self, rows, episodes, proof, seed):
        import numpy as np
        self.rng = random.Random(seed)
        self.rows, self.episodes = rows, episodes
        self.events = transition_events(rows, episodes)
        if not self.events['close'] or not self.events['open']:
            raise ValueError('Training sources need actual close and open target changes')
        self.by_source = defaultdict(list)
        self.holds = defaultdict(list)
        self.motion = defaultdict(list)
        self.pending_schedule = []
        self.pending_events = {'close': [], 'open': []}
        self.fourth_index = 0
        self.category_counts = {name: 0 for name in ('close', 'open', 'motion', 'hold', 'all')}
        self.event_counts = {kind: {str(e['index']): {'total': 0, 'at': 0} for e in values}
                             for kind, values in self.events.items()}
        sources = {str(s['source_index']): s for s in proof['sources']}
        movement = []
        for indices in episodes.values():
            for i in indices:
                source = rows[i]['_physical_source']
                self.by_source[source].append(i)
                state, action = rows[i]['observation.state'], rows[i]['action']
                size = float(np.linalg.norm(np.asarray(action[:6])-state[:6]))
                movement.append((i, source, size))
                grid = rows[i]['_grid_time']
                if any(float(h['start_monotonic']) <= grid <= float(h['end_monotonic'])
                       for h in sources[source]['hold_records']):
                    self.holds[source].append(i)
        cutoff = float(np.quantile([x[2] for x in movement], .75))
        for i, source, size in movement:
            if size >= cutoff and size > 0:
                self.motion[source].append(i)
        if not self.motion or not self.holds:
            raise ValueError('Measured movement and recorded hover frames must both be present')
        self.motion_cutoff_radians = cutoff

    def _pick_source_frame(self, table):
        source = self.rng.choice(sorted(table))
        return self.rng.choice(table[source])

    def draw(self):
        if not self.pending_schedule:
            self.pending_schedule = list(self.SCHEDULE)
            self.rng.shuffle(self.pending_schedule)
        return self.draw_category(self.pending_schedule.pop())

    def draw_category(self, category):
        if category not in self.category_counts:
            raise ValueError('Unknown sampler category')
        self.category_counts[category] += 1
        if category in ('close', 'open'):
            queue = self.pending_events[category]
            if not queue:
                queue.extend((i, exact) for i in range(len(self.events[category])) for exact in (True, False))
                self.rng.shuffle(queue)
            event_index, exact = queue.pop()
            event = self.events[category][event_index]
            delta = 0 if exact else self.rng.choice((-3, -2, -1, 1, 2, 3))
            indices = self.episodes[event['episode']]
            pos = min(len(indices)-1, max(0, event['position']+delta))
            counts = self.event_counts[category][str(event['index'])]
            counts['total'] += 1
            counts['at'] += int(pos == event['position'])
            return indices[pos], {'category': category, 'event_index': event['index'],
                                  'source': event['source'], 'offset_frames': pos-event['position']}
        table = {'motion': self.motion, 'hold': self.holds, 'all': self.by_source}[category]
        index = self._pick_source_frame(table)
        return index, {'category': category, 'source': self.rows[index]['_physical_source']}

    def draw_balanced(self, count):
        if count < 4 or count % 4:
            raise ValueError('Balanced effective batch must be a positive multiple of four')
        result = []
        for _ in range(count // 4):
            fourth = self.FOURTH_SCHEDULE[self.fourth_index % len(self.FOURTH_SCHEDULE)]
            self.fourth_index += 1
            group = [self.draw_category(kind) for kind in ('close', 'open', 'motion', fourth)]
            self.rng.shuffle(group)
            result.extend(group)
        return result

    def getstate(self):
        return {'version': self.VERSION, 'rng': self.rng.getstate(),
                'pending_schedule': list(self.pending_schedule), 'pending_events': copy.deepcopy(self.pending_events),
                'category_counts': dict(self.category_counts), 'event_counts': copy.deepcopy(self.event_counts),
                'fourth_index': self.fourth_index}

    def setstate(self, state):
        if not isinstance(state, dict) or state.get('version') != self.VERSION:
            raise ValueError('Sampler state belongs to a different sampling strategy')
        if set(state['event_counts']) != set(self.event_counts) or any(
                set(state['event_counts'][k]) != set(self.event_counts[k]) for k in self.event_counts):
            raise ValueError('Sampler events differ from the resumed dataset')
        self.rng.setstate(state['rng'])
        self.pending_schedule = list(state['pending_schedule'])
        self.pending_events = {k: list(v) for k, v in state['pending_events'].items()}
        self.category_counts = dict(state['category_counts'])
        self.event_counts = copy.deepcopy(state['event_counts'])
        self.fourth_index = int(state['fourth_index'])

    def report(self):
        return {'version': self.VERSION,
                'balanced_group': ['close', 'open', 'motion', 'all_or_recorded_hold'],
                'fourth_rotation': list(self.FOURTH_SCHEDULE),
                'category_counts': dict(self.category_counts), 'events': self.events,
                'event_counts': self.event_counts, 'recorded_hold_frames': sum(map(len, self.holds.values())),
                'recorded_hold_frames_by_source': {k: len(v) for k, v in self.holds.items()},
                'motion_frames': sum(map(len, self.motion.values())),
                'motion_quantile75_cutoff_radians': self.motion_cutoff_radians}


def anchor_coverage(rows, episodes, seen):
    """Report eligible versus actually sampled anchors and target changes."""
    all_indices = {i for ep in episodes.values() for i in ep}
    tail = {i for ep in episodes.values() for i in ep[-min(CHUNK - 1, len(ep)):]}
    closes, opens = set(), set()
    for ep in episodes.values():
        for left, right in zip(ep, ep[1:]):
            difference = rows[right]['action'][6] - rows[left]['action'][6]
            if difference < -1e-5:
                closes.add(right)
            elif difference > 1e-5:
                opens.add(right)
    seen = set(seen) & all_indices
    supervised_targets = set()
    for ep in episodes.values():
        for offset, index in enumerate(ep):
            if index in seen:
                supervised_targets.update(ep[offset:offset + CHUNK])
    return {'eligible_anchors': len(all_indices), 'sampled_unique_anchors': len(seen),
            'eligible_tail_anchors': len(tail), 'sampled_unique_tail_anchors': len(tail & seen),
            'supervised_unique_target_frames': len(supervised_targets),
            'eligible_closing_target_changes': len(closes), 'sampled_closing_target_changes': len(closes & seen),
            'eligible_opening_target_changes': len(opens), 'sampled_opening_target_changes': len(opens & seen),
            'supervised_closing_target_changes': len(closes & supervised_targets),
            'supervised_opening_target_changes': len(opens & supervised_targets),
            'tail_strategy': 'all_anchors_eligible_real_targets_preserved_padding_masked_in_attention_and_loss'}


@contextmanager
def masked_suffix_attention(model, valid):
    """Make LeRobot's suffix padding mask visible to its native attention.

    The suffix is embedded once outside the checkpointed transformer layers;
    the resulting attention mask remains part of checkpoint recomputation.
    """
    original = model.embed_suffix
    def masked(noisy_actions, timestep):
        embeddings, pad, ar, cond = original(noisy_actions, timestep)
        if tuple(pad.shape) != tuple(valid.shape):
            raise ValueError('Suffix mask shape differs from action chunk')
        return embeddings, pad & valid.to(device=pad.device, dtype=pad.dtype), ar, cond
    model.embed_suffix = masked
    try:
        yield
    finally:
        model.embed_suffix = original


@contextmanager
def action_clock_condition(policy, batch):
    """Give the action expert causal timing in unused action input channels."""
    clock = batch.get('observation.cup_action_clock')
    if clock is None:
        yield
        return
    import torch

    model = policy.model
    original = model.embed_suffix
    fixed = batch.get('observation.fixed_cup_action_clock', False)

    def conditioned(noisy_actions, timestep):
        if noisy_actions.shape[0] != clock.shape[0] or noisy_actions.shape[-1] < ACTION_DIM + clock.shape[-1]:
            raise ValueError('Cup clock does not fit the padded action input')
        time = torch.arange(1, noisy_actions.shape[1] + 1, device=noisy_actions.device) / FPS
        known = clock[:, 1:2]
        remaining = ((clock[:, :1] - time) / 2.).clamp(-1., 1.) * known
        flags = clock[:, None, 1:].expand(-1, noisy_actions.shape[1], -1)
        channels = 3. * torch.cat((remaining.unsqueeze(-1), flags), dim=-1).to(noisy_actions)
        end = ACTION_DIM + clock.shape[-1]
        return original(torch.cat((noisy_actions[..., :ACTION_DIM],
                                   channels if fixed else noisy_actions[..., ACTION_DIM:end] + channels,
                                   noisy_actions[..., end:]), dim=-1), timestep)

    model.embed_suffix = conditioned
    try:
        yield
    finally:
        model.embed_suffix = original


def predict_action_chunk(policy, batch):
    with action_clock_condition(policy, batch):
        return policy.predict_action_chunk(batch)


def cup_boundary_penalty(clean_error, boundary, clock, action_scale, weights, *, support_focus=False,
                         coupled_support=False, setdown_age=None, before_setdown=None, include_xyz=True):
    """Measure the one-step denoising error in mm and gripper percentage points."""
    import torch
    import torch.nn.functional as F

    target_delta_mm, target_opening, normal, gap_mm, hover, returning, contact, actual_open, raw_remaining = boundary
    prefix = clean_error.shape[1]
    ticks = torch.arange(1, prefix + 1, device=clean_error.device)
    normal_error_mm = ((clean_error[..., :3] * action_scale[:3]) * normal[:, None, :]).sum(-1) * ticks * 1000.
    opening_error_pct = clean_error[..., 6] * action_scale[6] * 100.
    target_gap_mm = gap_mm[:, None] + target_delta_mm
    platform = (hover[:, None] & contact[:, None] & (target_delta_mm.abs() <= 1.25)
                & (clock[:, 0:1] - ticks / FPS > 0.) & (clock[:, 1:2] == 1.))
    if support_focus:
        platform = (hover[:, None] & contact[:, None] & (target_delta_mm.abs() <= 1.25)
                    & (raw_remaining[:, None] - ticks / FPS > 0.))
    setdown = returning[:, None] & contact[:, None] & (target_gap_mm.abs() <= 1.5)
    held = (hover | returning)[:, None] & contact[:, None] & (target_opening < .88)
    real_release = returning[:, None] & contact[:, None] & (target_opening >= .98)
    real_close = actual_open[:, None] & (target_opening < .88)
    open_approach = actual_open[:, None] & (target_opening >= .98)

    def huber_excess(error, margin, scale):
        excess = (error - margin).clamp_min(0.) / scale
        return F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none')

    z = (platform * huber_excess(-normal_error_mm, .5, 1.5)
         + setdown * huber_excess(normal_error_mm, .5, 1.5))
    grip = ((held | real_close) * huber_excess(opening_error_pct, 2., 5.)
            + (real_release | open_approach) * huber_excess(-opening_error_pct, 2., 5.))
    penalty = 2. * z * weights[..., 2] + 2. * grip * weights[..., 6]
    if support_focus:
        near_support = returning[:, None] & contact[:, None] & (gap_mm[:, None] <= 18.)
        grip_at_support = huber_excess(opening_error_pct, 1., 5.) * (near_support & held)
        grip_at_support += huber_excess(-opening_error_pct, 1., 5.) * (near_support & real_release)
        # Use the demonstrated height and switch time as separate targets.
        # A late height prediction must not be rewarded for simply staying closed.
        z_at_support = setdown * huber_excess(normal_error_mm.abs(), .4, 1.0)
        z_at_platform = platform * huber_excess(-normal_error_mm, .35, 1.0)
        penalty += 6. * (z_at_support + z_at_platform) * weights[..., 2] + 6. * grip_at_support * weights[..., 6]
    if coupled_support:
        # On a recorded hold tick, an unsafe proposed opening is corrected
        # through the gripper output, not by teaching an earlier descent.
        proposed_gap_mm = target_gap_mm + normal_error_mm
        height_excess = (proposed_gap_mm.detach() - 1.5).clamp(0., 6.)
        unsafe_open = (near_support & held) * height_excess * huber_excess(opening_error_pct, 1., 5.)
        # At an actual release the pose must reach the recorded height and the
        # gripper must open. Match the preceding descent as well as the endpoint.
        return_path = near_support * huber_excess(normal_error_mm.abs(), .4, 1.5)
        due_close = real_close & (ticks <= 4)
        close_now = due_close * huber_excess(opening_error_pct, 1., 5.)
        penalty += (5. * unsafe_open + 8. * close_now) * weights[..., 6] + 6. * return_path * weights[..., 2]
        penalty += 4. * open_approach * huber_excess(-opening_error_pct, 2., 5.) * weights[..., 6]
        # Match the original short trajectory in all three translation axes,
        # including its stationary pickup and hover points.
        if include_xyz:
            translation_error_mm = clean_error[..., :3] * action_scale[:3] * ticks[:, None] * 1000.
            translation_error = translation_error_mm.square().sum(-1).add(1e-8).sqrt()
            penalty += 6. * huber_excess(translation_error, .25, 1.) * weights[..., 0]
    if setdown_age is not None:
        # Teach timely opening on actual command-bearing observations without
        # giving the arm an incentive to descend to satisfy the gripper loss.
        ready = setdown & (setdown_age[:, None] >= .30)
        # A command arriving a fraction into an already stable action window
        # should supervise prompt opening, without inventing an earlier target.
        due = ready & real_release
        penalty += 28. * due * huber_excess(-opening_error_pct, 1., 3.) * weights[..., 6]
        # Poses associated with the demonstrated open have nearly stopped even
        # in XY; enforce that in the policy before reaching the controller.
        displacement_error_mm = clean_error[..., :3] * action_scale[:3] * ticks[:, None] * 1000.
        settled_error_mm = displacement_error_mm.square().sum(-1).add(1e-8).sqrt()
        penalty += 8. * due * huber_excess(settled_error_mm, .2, .6) * weights[..., 2]
    if before_setdown is not None:
        # Give extra weight to the recorded held command while the original
        # feedback has not yet reached stable support.
        predicted_opening = target_opening * 100. + opening_error_pct
        penalty += 16. * before_setdown * huber_excess(predicted_opening - 76., 2., 4.) * weights[..., 6]
    return penalty.sum()


def masked_flow_loss(policy, batch, valid, *, first_action_weight=1., gripper_weight=1.,
                     prefix_steps=1, prefix_weight=1., vertical_weight=1., cup_action_scale=None,
                     support_focus=False, coupled_support=False, xyz_loss_mode='clean', loss_parts=None,
                     backward_scale=None):
    """Compute masked losses; optionally backpropagate flow before sampled XYZ."""
    import torch
    if valid.ndim != 2 or valid.shape[1] != CHUNK or not bool(valid[:, 0].all()):
        raise ValueError('Invalid target mask')
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError('Valid targets must precede suffix padding')
    if xyz_loss_mode not in ('clean', 'terminal'):
        raise ValueError('Unknown XYZ supervision mode')
    if backward_scale is not None and (xyz_loss_mode != 'terminal'
            or not math.isfinite(backward_scale) or backward_scale <= 0.):
        raise ValueError('Separate flow backward requires terminal XYZ and a positive microbatch scale')
    if xyz_loss_mode == 'terminal' and (not coupled_support or cup_action_scale is None
                                       or batch.get('observation.fixed_cup_action_clock') is not True):
        raise ValueError('Terminal XYZ supervision requires fixed cup conditioning and coupled support')
    images, image_masks = policy._preprocess_images(batch)
    actions = policy.prepare_action(batch)
    noise = policy.model.sample_noise(actions.shape, actions.device)
    sample_time = policy.model.sample_time(actions.shape[0], actions.device)
    velocity = []
    hook = (policy.model.action_out_proj.register_forward_hook(lambda _m, _a, out: velocity.append(out))
            if cup_action_scale is not None else None)
    try:
        with action_clock_condition(policy, batch), masked_suffix_attention(policy.model, valid):
            losses = policy.model.forward(images, image_masks, batch['observation.language.tokens'],
                                          batch['observation.language.attention_mask'], actions, noise, sample_time)
    finally:
        if hook is not None:
            hook.remove()
    losses = losses[:, :, :ACTION_DIM]
    if losses.shape[:2] != valid.shape:
        raise ValueError('Loss shape differs from target mask')
    if not (math.isfinite(first_action_weight) and first_action_weight > 0
            and math.isfinite(gripper_weight) and gripper_weight > 0):
        raise ValueError('Training loss weights must be positive and finite')
    temporal = losses.new_ones((1, CHUNK, 1))
    temporal[:, :prefix_steps] = prefix_weight
    temporal[:, 0] = first_action_weight
    dimensions = losses.new_ones((1, 1, ACTION_DIM))
    dimensions[:, :, 2] = vertical_weight
    dimensions[:, :, 6] = gripper_weight
    weights = valid.to(device=losses.device, dtype=losses.dtype).unsqueeze(-1)*temporal*dimensions
    supervision = weights
    if 'cup_settled_wait_mask' in batch or 'cup_pickup_wait_mask' in batch:
        supervision = weights.clone()
        for key in ('cup_settled_wait_mask', 'cup_pickup_wait_mask'):
            if key in batch:
                supervision[..., 6].masked_fill_(batch[key], 0.)
    numerator = (losses * supervision).sum()
    if loss_parts is not None:
        loss_parts['flow'] = float((numerator / weights.sum()).detach())
    if cup_action_scale is not None:
        clean_error = sample_time[:, None, None] * (
            (noise - actions)[:, :prefix_steps, :ACTION_DIM] - velocity[0][:, :prefix_steps, :ACTION_DIM].float())
        boundary_penalty = cup_boundary_penalty(clean_error, batch['cup_boundary'],
            batch['observation.cup_action_clock'], cup_action_scale, supervision[:, :prefix_steps],
            support_focus=support_focus, coupled_support=coupled_support, setdown_age=batch.get('cup_setdown_age'),
            before_setdown=batch.get('cup_before_setdown'), include_xyz=xyz_loss_mode == 'clean')
        numerator = numerator + boundary_penalty
        if loss_parts is not None:
            loss_parts['boundary'] = float((boundary_penalty / weights.sum()).detach())
            if coupled_support and xyz_loss_mode == 'clean':
                from stream_terminal_loss import terminal_xyz_penalty
                clean_xyz = terminal_xyz_penalty(clean_error.detach(), cup_action_scale,
                                                 supervision[:, :prefix_steps])
                loss_parts['xyz_clean'] = float(clean_xyz / weights.sum())
                loss_parts['boundary'] -= loss_parts['xyz_clean']
        if xyz_loss_mode == 'terminal':
            from stream_terminal_loss import sample_terminal_actions, terminal_xyz_penalty
            if backward_scale is not None:
                (numerator / weights.sum() * backward_scale).backward()
                numerator = numerator.detach()
                velocity.clear()
            terminal = sample_terminal_actions(policy, batch, noise)
            terminal_error = terminal[:, :prefix_steps] - actions[:, :prefix_steps, :ACTION_DIM]
            terminal_penalty = terminal_xyz_penalty(terminal_error, cup_action_scale, supervision[:, :prefix_steps])
            numerator = numerator + terminal_penalty
            if loss_parts is not None:
                loss_parts['xyz_terminal'] = float((terminal_penalty / weights.sum()).detach())
    return numerator / weights.sum()


def loss_normalizer(counts, first_action_weight, gripper_weight, *, prefix_steps=1, prefix_weight=1., vertical_weight=1.):
    """Real target weights per microbatch, for exact accumulated loss scaling."""
    if not counts or any(not 1 <= n <= CHUNK for n in counts):
        raise ValueError('Each accumulated sample must contain real target frames')
    return sum(first_action_weight + (min(n, prefix_steps) - 1) * prefix_weight
               + max(0, n - prefix_steps) for n in counts) * (ACTION_DIM - 2 + vertical_weight + gripper_weight)


def resolve_xyz_loss_mode(requested, *, parent_report=None, resume=False):
    parent = parent_report.get('xyz_loss_mode', 'clean') if parent_report is not None else 'clean'
    mode = requested if requested is not None else parent
    if mode not in ('clean', 'terminal'):
        raise ValueError('Unknown XYZ supervision mode')
    if resume and mode != parent:
        raise ValueError('Resume requires the saved XYZ supervision mode')
    return mode


def resolve_action_steps(requested, *, parent_steps=None, resume=False):
    """Inherit a sealed prefix on resume; allow explicit changes only for new runs."""
    if parent_steps is not None:
        checkpoints.configured_action_steps({'n_action_steps': parent_steps}, CHUNK)
    if resume and parent_steps is None:
        raise ValueError('Resume requires the sealed parent n_action_steps')
    if requested is None:
        return CHUNK if parent_steps is None else parent_steps
    checkpoints.configured_action_steps({'n_action_steps': requested}, CHUNK)
    if resume and requested != parent_steps:
        raise ValueError('Resume n_action_steps must match the sealed parent configuration')
    return requested


def resolve_image_profile(requested, *, parent_profile=None, resume=False):
    """A resume must preserve its sealed views; a warm start may choose anew."""
    if parent_profile is not None:
        checkpoints.image_profile_metadata(parent_profile)
    if resume and parent_profile is None:
        raise ValueError('Resume requires the sealed parent image profile')
    name = requested if requested is not None else (parent_profile or FULL_IMAGE_PROFILE)
    checkpoints.image_profile_metadata(name)
    if resume and name != parent_profile:
        raise ValueError('Resume image profile must match the sealed parent configuration')
    return name


def resolve_sampling_profile(requested, *, parent_report=None, resume=False):
    versions = {DEFAULT_SAMPLING_PROFILE: TransitionSampler.VERSION,
                APPROACH_SAMPLING_PROFILE: conditioned_sampling.APPROACH_OPEN_VERSION,
                EARLY_SWITCH_SAMPLING_PROFILE: conditioned_sampling.EARLY_SWITCH_CONTEXT_VERSION,
                CUP_TIMING_SAMPLING_PROFILE: conditioned_sampling.CUP_TIMING_SAMPLING_VERSION,
                CUP_CLOCK_SAMPLING_PROFILE: conditioned_sampling.CUP_CLOCK_SAMPLING_VERSION,
                CUP_EVENT_SAMPLING_PROFILE: conditioned_sampling.CUP_EVENT_SAMPLING_VERSION,
                CUP_BOUNDARY_SAMPLING_PROFILE: conditioned_sampling.CUP_BOUNDARY_SAMPLING_VERSION,
                CUP_SUPPORT_SAMPLING_PROFILE: conditioned_sampling.CUP_SUPPORT_SAMPLING_VERSION,
                CUP_COUPLED_SAMPLING_PROFILE: conditioned_sampling.CUP_COUPLED_SAMPLING_VERSION,
                UNIFORM_SAMPLING_PROFILE: UniformFrameSampler.VERSION,
                UNIFORM_SWITCH_PROFILE: UniformSwitchSampler.VERSION,
                UNIFORM_SWITCH_HOLD_PROFILE: UniformSwitchHoldSampler.VERSION}
    parent = None
    if parent_report is not None:
        parent = parent_report.get('sampling_profile')
        if parent is None:
            parent = next((name for name, version in versions.items()
                           if parent_report.get('sampling') == version), None)
    name = requested if requested is not None else (parent or DEFAULT_SAMPLING_PROFILE)
    if name not in versions:
        raise ValueError('Unknown sampling profile')
    if resume and (parent is None or name != parent
                   or parent_report.get('sampling') != versions[name]):
        raise ValueError('Resume requires the same sampler; use --initialize for a changed profile')
    return name


def resolve_observation_context_profile(requested, *, parent_profile=None, resume=False):
    if resume and parent_profile is None:
        raise ValueError('Resume requires the sealed parent observation context profile')
    name = requested if requested is not None else (parent_profile or observation_history.NONE_PROFILE)
    observation_history.context_metadata(name)
    if resume and name != parent_profile:
        raise ValueError('Resume requires the same observation context profile')
    return name


def configure_observation_context(config, profile):
    observation_history.context_metadata(profile)
    if profile in observation_history.HISTORY_PROFILES:
        config.tokenizer_max_length = prompt_integrity.MAX_PROMPT_TOKENS


def attach_observation_history(rows, episodes):
    """Attach same-episode history using the 6 Hz dataset task clock."""
    for indices in episodes.values():
        samples = [(int(_scalar(rows[i]['frame_index'])) / FPS, rows[i]['observation.state']) for i in indices]
        for index, (timestamp, _) in zip(indices, samples):
            history = observation_history.select_observation_history(samples, timestamp)
            rows[index]['_observation_history'] = [
                None if history[offset] is None else
                {'age_s': history[offset]['age_s'], 'state': history[offset]['state']}
                for offset in observation_history.HISTORY_OFFSETS_S]


def attach_physical_observation_history(rows, episodes, proof):
    """Select causal history from each original continuous capture journal."""
    timing, feedback = load_record_helpers()
    for indices in episodes.values():
        source_index = int(rows[indices[0]]['_physical_source'])
        source = proof['sources'][source_index]
        journal = Path(source['trial']) / 'continuous/capture.jsonl'
        expected = source.get('capture_journal_sha256')
        synthetic = (proof.get('training_data_kind') == 'synthetic_diagnostic'
                     and source.get('data_provenance', {}).get('training_data_kind') == 'synthetic_diagnostic')
        if expected is None and not synthetic:
            raise ValueError(f'Physical history requires a capture journal receipt for source {source_index}')
        raw = journal.read_bytes()
        if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'Physical history capture journal differs from provenance for source {source_index}')
        captures = [json.loads(line) for line in raw.splitlines() if line.strip()]
        samples, frames = [], {}
        first_time = rows[indices[0]]['_camera_time']
        for capture in captures:
            timestamp, basis = timing.camera_clock(capture['camera'])
            if basis != timing.MONOTONIC:
                raise ValueError('Physical history requires recorded camera monotonic times')
            robot = capture['robot']
            valid = feedback.gripper_feedback_valid(robot)
            state = [*[math.radians(q) for q in robot['joints']],
                     robot['gripper_position'] / 100. if valid else 0., float(valid)]
            frames[capture['frame_index']] = timestamp
            if timestamp >= first_time:
                samples.append((timestamp, state))
        for index in indices:
            row = rows[index]
            timestamp = row['_camera_time']
            if abs(frames[row['_source_capture_frame']] - timestamp) > 1e-5:
                raise ValueError('Physical history journal does not match the observation source')
            history = observation_history.select_observation_history(samples, timestamp)
            row['_physical_observation_history'] = [
                None if history[offset] is None else
                {'age_s': history[offset]['age_s'], 'state': history[offset]['state']}
                for offset in observation_history.HISTORY_OFFSETS_S]


def attach_initial_cup(rows, episodes, proof):
    """Use the rim observed before motion, never a cup or target from a future frame."""
    observations = []
    for indices in episodes.values():
        source_index = int(rows[indices[0]]['_physical_source'])
        source = proof['sources'][source_index]
        root = Path(source['trial'])
        control = json.loads((root / 'control-manifest.json').read_text())
        raw = (root / 'geometry.json').read_bytes()
        if (control['scope_sha256'] != source['source_scope_sha256']
                or control['files']['geometry.json']['sha256'] != hashlib.sha256(raw).hexdigest()):
            raise ValueError(f'Initial cup observation differs from the archive for source {source_index}')
        geometry = json.loads(raw)
        uv = observation_history.initial_cup_pixel(geometry['cup_uv_selected_by_llm'])
        observations.append({'source_index': source_index, 'initial_cup_uv': uv,
                             'source_observation': geometry['source_observation']})
        for i in indices:
            rows[i]['_initial_cup_uv'] = uv
    return observations


def make_sampler(profile, rows, episodes, proof, seed):
    if profile == DEFAULT_SAMPLING_PROFILE:
        cls = TransitionSampler
    elif profile == APPROACH_SAMPLING_PROFILE:
        cls = conditioned_sampling.approach_open_sampler_class(TransitionSampler)
    elif profile == EARLY_SWITCH_SAMPLING_PROFILE:
        cls = conditioned_sampling.early_switch_context_sampler_class(TransitionSampler)
    elif profile in (CUP_TIMING_SAMPLING_PROFILE, CUP_CLOCK_SAMPLING_PROFILE, CUP_EVENT_SAMPLING_PROFILE,
                     CUP_BOUNDARY_SAMPLING_PROFILE, CUP_SUPPORT_SAMPLING_PROFILE, CUP_COUPLED_SAMPLING_PROFILE):
        support = profile in (CUP_SUPPORT_SAMPLING_PROFILE, CUP_COUPLED_SAMPLING_PROFILE)
        boundaries = profile == CUP_BOUNDARY_SAMPLING_PROFILE or support
        cls = conditioned_sampling.cup_timing_sampler_class(TransitionSampler,
            focus_hold_return=profile != CUP_TIMING_SAMPLING_PROFILE,
            camera_events=profile == CUP_EVENT_SAMPLING_PROFILE or boundaries,
            camera_boundaries=boundaries, support_focus=support, coupled_support=profile == CUP_COUPLED_SAMPLING_PROFILE)
    elif profile == UNIFORM_SAMPLING_PROFILE:
        cls = UniformFrameSampler
    elif profile == UNIFORM_SWITCH_PROFILE:
        cls = UniformSwitchSampler
    elif profile == UNIFORM_SWITCH_HOLD_PROFILE:
        cls = UniformSwitchHoldSampler
    else:
        raise ValueError('Unknown sampling profile')
    return cls(rows, episodes, proof, seed)


def make_config(n_action_steps=CHUNK, *, adaptation=LORA_ADAPTATION):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    checkpoints.configured_action_steps({'n_action_steps': n_action_steps}, CHUNK)
    if adaptation not in ADAPTATION_MODES:
        raise ValueError('Unknown adaptation mode')
    features = {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in legacy.MODEL_IMAGE_KEYS}
    features['observation.state'] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
    return PI05Config(input_features=features,
                      output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
                      dtype='bfloat16', device='cpu', chunk_size=CHUNK, n_action_steps=n_action_steps,
                      gradient_checkpointing=True, freeze_vision_encoder=adaptation != VISION_EXPERT_ADAPTATION,
                      train_expert_only=False,
                      push_to_hub=False, use_relative_actions=False)


def is_expert_base(name):
    return '.gemma_expert.model.' in name and '.lora_' not in name


_VISION_ADAPTER = re.compile(
    r'.*\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)'
    r'\.self_attn\.([qv])_proj\.lora_([AB])(?:\.default)?\.weight'
)
_VISION_LAYER = re.compile(
    r'.*\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)'
)


def is_vision_adapter(name):
    return _VISION_ADAPTER.fullmatch(name) is not None


def vision_adapter_inventory(names):
    """Require both LoRA matrices for q and v in every represented vision layer."""
    layers = defaultdict(set)
    for name in names:
        match = _VISION_ADAPTER.fullmatch(name)
        if match:
            layers[int(match[1])].add((match[2], match[3]))
    expected = {(projection, matrix) for projection in ('q', 'v') for matrix in ('A', 'B')}
    if not layers or any(items != expected for items in layers.values()):
        raise ValueError('Vision LoRA must include q/v A/B tensors for every adapted layer')
    return sorted(layers)


def trainable_parameters(policy, adaptation=LORA_ADAPTATION):
    """Select LoRA or full action expert, plus complete action projections."""
    if adaptation not in ADAPTATION_MODES:
        raise ValueError('Unknown adaptation mode')
    parameters, language, expert, vision, projections = [], 0, 0, [], set()
    for name, parameter in policy.named_parameters():
        adapter = '.lora_' in name
        projection = name.startswith(PROJECTION_PREFIXES)
        language_adapter = adapter and '.paligemma.model.language_model.' in name
        expert_adapter = adapter and '.gemma_expert.model.' in name
        vision_adapter = adapter and is_vision_adapter(name)
        if adaptation == VISION_EXPERT_ADAPTATION and adapter and '.vision_tower.' in name and not vision_adapter:
            raise ValueError(f'Unexpected vision LoRA tensor: {name}')
        full_expert = adaptation in FULL_EXPERT_MODES and is_expert_base(name)
        permitted_adapter = (language_adapter or (expert_adapter and adaptation == LORA_ADAPTATION)
                             or (vision_adapter and adaptation == VISION_EXPERT_ADAPTATION))
        parameter.requires_grad_(permitted_adapter or projection or full_expert)
        # Keep the inherited expert LoRA in FP32 even when frozen: its constant
        # contribution is part of the common initialization and saved adapter.
        if adapter:
            parameter.data = parameter.data.float()
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
            parameters.append(parameter)
            language += int(language_adapter)
            expert += int(expert_adapter or full_expert)
            if vision_adapter:
                vision.append(name)
            if projection:
                projections.add(name)
    required = {f'{p}{s}' for p in PROJECTION_PREFIXES for s in ('weight', 'bias')}
    if not language or not expert or projections != required:
        raise ValueError('Missing language adaptation, action expert or full action projections')
    if adaptation == VISION_EXPERT_ADAPTATION:
        adapted_layers = set(vision_adapter_inventory(vision))
        encoder_layers = {int(match[1]) for name, _ in policy.named_modules()
                          if (match := _VISION_LAYER.fullmatch(name))}
        if not encoder_layers or adapted_layers != encoder_layers:
            raise ValueError('Vision LoRA does not cover every SigLIP encoder layer')
    return parameters


def projection_state(policy):
    return {name: p.detach().cpu().contiguous() for name, p in policy.named_parameters()
            if name.startswith(PROJECTION_PREFIXES)}


def expert_state(policy):
    """The action expert's forward model, excluding its unused language head."""
    return {name: p.detach().cpu().contiguous() for name, p in policy.named_parameters() if is_expert_base(name)}


def adaptation_metadata(mode):
    if mode not in ADAPTATION_MODES:
        raise ValueError('Unknown adaptation mode')
    full = mode in FULL_EXPERT_MODES
    value = {'schema': 'fr5-stream-adaptation-v1', 'mode': mode,
            'language': 'q_v_lora', 'action_projections': 'full_fp32',
            'action_expert': ('full_model_fp32' if full else 'q_v_lora'),
            'inherited_expert_lora': ('frozen_and_saved' if full else 'trainable'),
            'unused_expert_language_head': 'frozen_not_used_by_action_forward'}
    if mode == VISION_EXPERT_ADAPTATION:
        value['vision'] = 'siglip_encoder_q_v_lora'
    return value


def vision_adapter_report(policy, mode, *, frozen=False):
    if mode != VISION_EXPERT_ADAPTATION:
        return None
    values = [(name, p) for name, p in policy.named_parameters() if is_vision_adapter(name)]
    layers = vision_adapter_inventory(name for name, _ in values)
    if any(p.requires_grad == frozen for _, p in values):
        raise ValueError('Vision LoRA trainability does not match the training mode')
    return {'scope': 'siglip_vision_encoder_self_attention_q_v', 'layers': layers,
            'tensor_count': len(values), 'trainable_parameters': sum(p.numel() for _, p in values if p.requires_grad),
            'frozen': frozen,
            'tensor_names': [name for name, _ in values]}


def restore_expert(policy, path):
    """Restore every full-expert FP32 tensor, rejecting any partial snapshot."""
    import torch
    from safetensors.torch import load_file
    expert = load_file(str(path), device='cpu')
    current = dict(policy.named_parameters())
    wanted = {name for name in current if is_expert_base(name)}
    if not wanted or set(expert) != wanted:
        raise ValueError('Saved full action-expert tensors do not match model')
    with torch.no_grad():
        for name, value in expert.items():
            if value.dtype != torch.float32 or current[name].dtype != torch.float32 or current[name].shape != value.shape:
                raise ValueError('Saved full action-expert shape or dtype differs')
            current[name].copy_(value)
            if not torch.equal(current[name].detach().cpu(), value):
                raise ValueError('Action-expert reload changed saved values')


def gradient_summary(policy):
    """Confirm the intended components actually receive gradients."""
    import torch
    groups = defaultdict(list)
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        if is_expert_base(name):
            kind = 'action_expert'
        elif '.gemma_expert.model.' in name and '.lora_' in name:
            kind = 'expert_lora'
        elif '.paligemma.model.language_model.' in name and '.lora_' in name:
            kind = 'language_lora'
        elif is_vision_adapter(name):
            kind = 'vision_lora'
        elif name.startswith(PROJECTION_PREFIXES):
            kind = 'action_projections'
        else:
            raise ValueError(f'Unexpected gradient on {name}')
        groups[kind].append(parameter.grad.detach().float().norm())
    return {name: {'tensors': len(values), 'norm': float(torch.stack(values).norm())}
            for name, values in groups.items()}


def reload_parameters(policy, checkpoint, *, allow_vision_upgrade=False, target_mode=None):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    root = Path(checkpoint)
    mode = checkpoints.read_adaptation(root)['mode']
    if allow_vision_upgrade and (mode != FULL_EXPERT_ADAPTATION or target_mode != VISION_EXPERT_ADAPTATION):
        raise ValueError('Vision adapter upgrade requires an explicitly initialized full-expert checkpoint')
    expected = get_peft_model_state_dict(policy)
    saved = load_file(str(root / 'adapter_model.safetensors'), device='cpu')
    if allow_vision_upgrade:
        vision_keys = {name for name in expected if is_vision_adapter(name)}
        vision_adapter_inventory(vision_keys)
        if not vision_keys or set(expected) - vision_keys != set(saved):
            raise ValueError('Upgrade requires every old adapter and exactly the new vision LoRA tensors')
        vision_initial = {name: expected[name].detach().cpu().clone() for name in vision_keys}
        for name, value in vision_initial.items():
            if not torch.isfinite(value).all() or ('.lora_B.' in name and torch.count_nonzero(value).item()):
                raise ValueError('New vision LoRA B tensors must be zero initialized')
        saved = {**saved, **vision_initial}
    if (set(expected) != set(saved) or any(expected[k].shape != saved[k].shape for k in expected)
            or (target_mode == VISION_EXPERT_ADAPTATION
                and any(expected[k].dtype != saved[k].dtype for k in expected))):
        raise ValueError('Adapter tensor keys or shapes differ')
    result = set_peft_model_state_dict(policy, saved, ignore_mismatched_sizes=False)
    if result.unexpected_keys or any('lora_' in key for key in result.missing_keys):
        raise ValueError('Adapter reload has missing/unexpected keys')
    if any(not torch.equal(v.detach().cpu(), saved[k]) for k, v in get_peft_model_state_dict(policy).items()):
        raise ValueError('Adapter reload changed saved values')
    projections = load_file(str(root / 'projections.safetensors'), device='cpu')
    current = dict(policy.named_parameters())
    wanted = {name for name in current if name.startswith(PROJECTION_PREFIXES)}
    if set(projections) != wanted:
        raise ValueError('Saved action projections do not match model')
    with torch.no_grad():
        for name, value in projections.items():
            if current[name].shape != value.shape:
                raise ValueError('Projection shape differs')
            current[name].copy_(value)
            if not torch.equal(current[name].detach().cpu(), value):
                raise ValueError('Projection reload changed saved values')
    if mode in FULL_EXPERT_MODES:
        restore_expert(policy, root / 'expert_model.safetensors')


def load_policy(config, base_weights, *, rank=32, checkpoint=None, adaptation=None, allow_vision_upgrade=False):
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import load_file
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    parent_mode = checkpoints.read_adaptation(checkpoint)['mode'] if checkpoint else None
    mode = adaptation or parent_mode or LORA_ADAPTATION
    if mode not in ADAPTATION_MODES:
        raise ValueError('Unknown adaptation mode')
    if allow_vision_upgrade and (not checkpoint or parent_mode != FULL_EXPERT_ADAPTATION
                                 or mode != VISION_EXPERT_ADAPTATION):
        raise ValueError('Vision upgrade must explicitly initialize from a full-expert checkpoint')
    if checkpoint and mode == VISION_EXPERT_ADAPTATION and parent_mode != mode and not allow_vision_upgrade:
        raise ValueError('Vision adapter upgrade requires explicit initialization')
    if checkpoint and parent_mode == VISION_EXPERT_ADAPTATION and mode != parent_mode:
        raise ValueError('Cannot silently discard saved vision adapters')
    if mode == VISION_EXPERT_ADAPTATION and (config.freeze_vision_encoder is not False
                                              or config.train_expert_only is not False):
        raise ValueError('Vision LoRA requires the vision path to remain trainable')
    policy = PI05Policy(config)
    original = load_file(str(base_weights), device='cpu')
    fixed = policy._fix_pytorch_state_dict_keys(original, config)
    mapped = {key if key.startswith('model.') else 'model.' + key: value for key, value in fixed.items()}
    policy.load_state_dict(mapped, strict=True)
    del original, fixed, mapped
    lora = (LoraConfig.from_pretrained(str(checkpoint), local_files_only=True) if checkpoint else
            LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0., bias='none',
                       target_modules=(VISION_EXPERT_LORA_TARGETS if mode == VISION_EXPERT_ADAPTATION
                                       else LORA_TARGETS)))
    if allow_vision_upgrade:
        if lora.target_modules != LORA_TARGETS or lora.init_lora_weights is not True:
            raise ValueError('Parent adapter configuration is not compatible with the vision upgrade')
        lora.target_modules = VISION_EXPERT_LORA_TARGETS
    if mode == VISION_EXPERT_ADAPTATION and lora.target_modules != VISION_EXPERT_LORA_TARGETS:
        raise ValueError('Vision checkpoint has incompatible LoRA targets')
    policy = inject_adapter_in_model(lora, policy)
    trainable_parameters(policy, mode)
    if checkpoint:
        reload_parameters(policy, checkpoint, allow_vision_upgrade=allow_vision_upgrade, target_mode=mode)
    return policy


def validate_compact_clocks(maps, proof, cuts, clock_map):
    """Verify compact task ticks against explicit, evidence-backed source jumps."""
    def same(a, b):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-5)
    if clock_map.get('schema') != 'fr5-stream-clock-map-v2' or clock_map.get('task_fps') != FPS:
        raise ValueError('Invalid compact clock map')
    episodes = defaultdict(list)
    for frame in maps:
        episodes[frame['episode_index']].append(frame)
    clocks = clock_map.get('episodes', [])
    if len(clocks) != len(episodes) or {c['episode_index'] for c in clocks} != set(episodes):
        raise ValueError('Compact clock map episode coverage differs')
    cut_by_id = {}
    for cut in cuts:
        identifier = cut.get('cut_id')
        if not isinstance(identifier, str) or not identifier or identifier in cut_by_id:
            raise ValueError('Missing or duplicate compact cut identifier')
        cut_by_id[identifier] = cut
    used_cuts = set()
    for clock in clocks:
        ep = clock['episode_index']
        frames = episodes[ep]
        source = frames[0]['source_index']
        if clock.get('source_index') != source or clock.get('task_fps') != FPS:
            raise ValueError('Compact source clock does not match episode')
        segments = clock.get('segments', [])
        covered = []
        for segment in segments:
            a, b = segment.get('task_start_step'), segment.get('task_end_step')
            if (type(a) is not int or type(b) is not int or not 0 <= a <= b < len(frames)
                    or segment.get('episode_index') != ep or segment.get('source_index') != source):
                raise ValueError('Invalid compact clock segment')
            left, right = frames[a], frames[b]
            if (segment.get('source_v1_start_frame') != left['source_v1_frame']
                    or segment.get('source_v1_end_frame') != right['source_v1_frame']
                    or not same(segment.get('source_start_monotonic'), left['grid_time_monotonic'])
                    or not same(segment.get('source_end_monotonic'), right['grid_time_monotonic'])):
                raise ValueError('Compact clock segment endpoints disagree')
            covered.extend(range(a, b+1))
            for i in range(a, b):
                if frames[i]['cut_after'] or frames[i+1]['source_v1_frame'] != frames[i]['source_v1_frame']+1:
                    raise ValueError('Compact clock segment crosses a cut')
        if covered != list(range(len(frames))):
            raise ValueError('Compact clock segments do not cover each frame exactly once')
        for i, frame in enumerate(frames):
            grid, target = frame['grid_time_monotonic'], frame['target_time_monotonic']
            if (frame.get('step') != i or frame.get('source_index') != source
                    or type(frame.get('source_v1_frame')) is not int
                    or not same(frame.get('task_time_seconds'), i/FPS)
                    or not same(frame.get('target_task_time_seconds'), (i+1)/FPS)):
                raise ValueError('Invalid compact task clock')
            is_cut = frame.get('cut_after')
            if type(is_cut) is not bool:
                raise ValueError('Compact cut flag is not boolean')
            if i+1 < len(frames) and not same(target, frames[i+1]['grid_time_monotonic']):
                raise ValueError('Compact target is not the next source observation')
            if not is_cut:
                if frame.get('cut_id') is not None or not same(target-grid, 1/FPS):
                    raise ValueError('Unmarked compact source-clock jump')
                if i+1 < len(frames) and frames[i+1]['source_v1_frame'] != frame['source_v1_frame']+1:
                    raise ValueError('Unmarked omitted source frames')
                continue
            identifier = frame.get('cut_id')
            if identifier not in cut_by_id or identifier in used_cuts or i+1 >= len(frames):
                raise ValueError('Compact jump has missing or reused evidence')
            used_cuts.add(identifier)
            cut, right = cut_by_id[identifier], frames[i+1]
            lsrc, rsrc = frame['source_v1_frame'], right['source_v1_frame']
            if (cut.get('episode_index') != ep or cut.get('source_index') != source
                    or cut.get('split') != frame['split'] or cut.get('left_frame') != frame['frame']
                    or cut.get('right_frame') != right['frame'] or cut.get('left_task_step') != i
                    or cut.get('right_task_step') != i+1 or cut.get('left_source_v1_frame') != lsrc
                    or cut.get('right_source_v1_frame') != rsrc or rsrc <= lsrc+1
                    or cut.get('removed_source_v1_frames') != list(range(lsrc+1, rsrc))
                    or not same(cut.get('left_source_time_monotonic'), grid)
                    or not same(cut.get('right_source_time_monotonic'), target)
                    or not same(cut.get('left_task_time_seconds'), i/FPS)
                    or not same(cut.get('right_task_time_seconds'), (i+1)/FPS)
                    or not same(cut.get('source_elapsed_seconds'), target-grid)
                    or not same(cut.get('removed_task_seconds'), (rsrc-lsrc-1)/FPS)):
                raise ValueError('Compact jump does not match frame/clock evidence')
            previous, following, evidence = cut.get('previous_command', {}), cut.get('next_command', {}), cut.get('feedback', {})
            if (not isinstance(previous.get('completed_monotonic'), (int, float))
                    or not isinstance(following.get('issued_monotonic'), (int, float))
                    or not math.isfinite(previous['completed_monotonic'])
                    or not math.isfinite(following['issued_monotonic'])
                    or previous['completed_monotonic'] > grid+1e-5
                    or following['issued_monotonic'] < target-1e-5
                    or any(grid < h['end_monotonic'] and h['start_monotonic'] < target
                           for h in proof['sources'][source]['hold_records'])
                    or cut.get('hold_intersections') != []):
                raise ValueError('Compact jump crosses a command or recorded hold')
            spans = evidence.get('joint_span_deg', [])
            grip_min, grip_max = evidence.get('gripper_min_percent'), evidence.get('gripper_max_percent')
            if (evidence.get('all_gripper_feedback_valid') is not True
                    or len(spans) != 6 or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                             or not 0 <= v <= .020001 for v in spans)
                    or not isinstance(evidence.get('feedback_samples'), int) or evidence['feedback_samples'] < 2
                    or evidence.get('lower_time_monotonic', math.inf) > grid+1e-5
                    or evidence.get('upper_time_monotonic', -math.inf) < target-1e-5
                    or not isinstance(grip_min, (int, float)) or not isinstance(grip_max, (int, float))
                    or not math.isfinite(grip_min) or not math.isfinite(grip_max)
                    or not 0 <= grip_max-grip_min <= 1.000001):
                raise ValueError('Compact jump lacks stationary feedback evidence')
    if used_cuts != set(cut_by_id):
        raise ValueError('Compact cut evidence is not used exactly once')


def validate_terminal_observations(root, proof, maps, terminal):
    """Bind raw post-command evidence to each source without making action labels."""
    import numpy as np
    from PIL import Image
    timing, feedback = load_record_helpers()
    root = Path(root).resolve()
    if (terminal.get('schema') != 'fr5-stream-terminal-observations-v1'
            or terminal.get('observation_only') is not True):
        raise ValueError('Missing observation-only terminal evidence')
    last_by_source = {m['source_index']: m for m in maps}
    records = terminal.get('sources', [])
    if (len(records) != len(proof['sources'])
            or {r.get('source_index') for r in records} != set(range(len(proof['sources'])))):
        raise ValueError('Terminal evidence does not cover every physical source')
    files = []
    for record in records:
        source = record['source_index']
        last = last_by_source[source]
        def same(a, b):
            return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-5)
        if (record.get('episode_index') != last['episode_index'] or record.get('split') != last['split']
                or record.get('last_training_frame') != last['frame']
                or not same(record.get('last_training_grid_time_monotonic'), last['grid_time_monotonic'])
                or not same(record.get('last_training_target_time_monotonic'), last['target_time_monotonic'])):
            raise ValueError('Terminal evidence does not match its last training frame')
        command = record.get('last_command', {})
        issued, completed = command.get('issued_monotonic'), command.get('completed_monotonic')
        if (not isinstance(issued, (int, float)) or not isinstance(completed, (int, float))
                or not math.isfinite(issued) or not math.isfinite(completed) or completed < issued
                or not isinstance(command.get('file_sha256'), str) or len(command['file_sha256']) != 64):
            raise ValueError('Terminal final command evidence is invalid')
        observations = record.get('tail_observations', [])
        if not observations:
            raise ValueError('Terminal source has no original tail observations')
        previous_index, previous_camera, previous_robot = -1, -math.inf, -math.inf
        for obs in observations:
            index = obs.get('source_capture_frame')
            cam, robot = obs.get('camera_time_monotonic'), obs.get('robot_time_monotonic')
            raw_camera, raw_robot = obs.get('camera_raw', {}), obs.get('robot_raw', {})
            raw_time, basis = timing.camera_clock(raw_camera)
            if (type(index) is not int or index <= previous_index
                    or not isinstance(cam, (int, float)) or not isinstance(robot, (int, float))
                    or not math.isfinite(cam) or not math.isfinite(robot)
                    or cam <= previous_camera or robot <= previous_robot
                    or max(cam, robot) <= last['grid_time_monotonic']
                    or basis != timing.MONOTONIC or not same(cam, raw_time)
                    or not same(robot, raw_robot.get('received_monotonic'))
                    or not same(obs.get('camera_after_command_seconds'), cam-completed)
                    or not same(obs.get('robot_after_command_seconds'), robot-completed)
                    or obs.get('both_after_last_command') is not (cam > completed and robot > completed)
                    or obs.get('action_label', 'missing') is not None):
                raise ValueError('Invalid terminal source time, feedback or action label')
            valid = feedback.gripper_feedback_valid(raw_robot)
            if obs.get('gripper_valid') is not valid:
                raise ValueError('Terminal gripper validity differs from real feedback')
            joints = np.asarray(raw_robot.get('joints'), dtype=np.float64)
            if joints.shape != (6,) or not np.isfinite(joints).all():
                raise ValueError('Terminal joint feedback is invalid')
            expected = np.asarray([*np.deg2rad(joints),
                                   raw_robot['gripper_position']/100. if valid else 0., float(valid)], dtype=np.float32)
            if not np.array_equal(np.asarray(obs.get('state_radians_fraction'), dtype=np.float32), expected):
                raise ValueError('Terminal state differs from real robot feedback')
            image = checkpoints.safe_file(root, obs.get('rgb_copy_file', ''))
            if checkpoints.digest(image) != obs.get('rgb_jpeg_sha256'):
                raise ValueError('Terminal source JPEG hash differs')
            with Image.open(image) as picture:
                if picture.format != 'JPEG' or picture.size != (640, 360):
                    raise ValueError('Terminal source JPEG format or dimensions differ')
            files.append(image)
            previous_index, previous_camera, previous_robot = index, cam, robot
        if (observations[-1].get('both_after_last_command') is not True
                or record.get('terminal_source_capture_frame') != previous_index):
            raise ValueError('Missing actual feedback after the last completed command')
    return files


def read_dataset(path):
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(path)
    meta = root / 'meta'
    proof = json.loads((meta / 'stream_provenance.json').read_text())
    info = json.loads((meta / 'info.json').read_text())
    schema = proof.get('schema')
    if (schema not in ('fr5-stream-retarget-export-v1', 'fr5-stream-retarget-compact-v2')
            or proof.get('policy_contract') != load_contract()[0].contract()
            or 'interpolation of real feedback' not in proof.get('joint_action_semantics', '')):
        raise ValueError('Dataset must contain measured-trajectory-derived stream targets')
    if proof.get('training_data_kind') not in ('captured', 'synthetic_diagnostic'):
        raise ValueError('Dataset must declare its actual source data kind')
    if info.get('fps') != FPS or proof.get('fps') != FPS:
        raise ValueError('Expected fixed 6 Hz task dataset')
    for key, dim in (('observation.state', STATE_DIM), ('action', ACTION_DIM)):
        if info.get('features', {}).get(key, {}).get('shape') != [dim]:
            raise ValueError(f'Invalid feature shape for {key}')
    for name in ('stream_source_frames.jsonl', 'stream_stats.json', 'stream_verification.json'):
        if not (meta / name).is_file():
            raise ValueError(f'Missing stream metadata: {name}')
    verified = json.loads((meta / 'stream_verification.json').read_text())
    if verified.get('status') != 'passed':
        raise ValueError('Dataset source/label/video integrity verification did not pass')
    sources = proof.get('sources', [])
    episode_metadata = proof.get('episodes', [])
    source_by_episode = {}
    for episode in episode_metadata:
        ep, source = int(episode['episode_index']), int(episode['source_index'])
        if ep in source_by_episode or not 0 <= source < len(sources):
            raise ValueError('Invalid source/episode mapping')
        if sources[source]['split'] != episode['split']:
            raise ValueError('Episode split does not match physical source')
        source_by_episode[ep] = (source, episode['split'], int(episode['length']))
    for source_index, source in enumerate(sources):
        if source.get('source_index') != source_index:
            raise ValueError('Invalid physical source ordering')
        review = (root / source['user_review_file']).resolve()
        if root.resolve() not in review.parents or checkpoints.digest(review) != source['user_review_sha256']:
            raise ValueError('User review file/hash mismatch')
        verdict = json.loads(review.read_text(encoding='utf-8'))
        if (verdict.get('schema') != 'fr5-user-review-v1' or verdict.get('reviewer') != 'user'
                or verdict.get('source') != 'direct_user_message' or verdict.get('verdict') != 'success'
                or verdict.get('scope_sha256') != source.get('source_scope_sha256')
                or verdict.get('summary_sha256') != source.get('source_summary_sha256')):
            raise ValueError('Each source requires a matching user-confirmed success')
    paths = sorted((root / 'data').rglob('*.parquet'))
    rows = [row for p in paths for row in pq.read_table(p).to_pylist()]
    rows.sort(key=lambda r: int(_scalar(r['index'])))
    with (meta / 'stream_source_frames.jsonl').open(encoding='utf-8') as stream:
        maps = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != len(maps) or len(rows) != proof.get('frames'):
        raise ValueError('Frame provenance count differs from dataset')
    compact_files = []
    if schema == 'fr5-stream-retarget-compact-v2':
        compact_files = [meta / name for name in ('stream_cuts.jsonl', 'stream_clock_map.json',
                                                  'stream_compaction_report.json', 'stream_terminal_observations.json')]
        if not all(p.is_file() for p in compact_files):
            raise ValueError('Missing compact source-clock or terminal-observation evidence')
        cuts = [json.loads(line) for line in compact_files[0].read_text().splitlines() if line.strip()]
        validate_compact_clocks(maps, proof, cuts, json.loads(compact_files[1].read_text()))
        if proof.get('terminal_observations_file') != 'meta/stream_terminal_observations.json':
            raise ValueError('Compact dataset does not declare terminal observations')
        terminal = json.loads(compact_files[3].read_text())
        compact_files += validate_terminal_observations(root, proof, maps, terminal)
        if (verified.get('terminal_observation_sources') != len(proof['sources'])
                or verified.get('terminal_tail_observations') != len(compact_files)-4):
            raise ValueError('Verified terminal source or frame count differs')
    for index, (row, mapping) in enumerate(zip(rows, maps)):
        ep = int(_scalar(row['episode_index']))
        if ep not in source_by_episode:
            raise ValueError('Frame has no physical source')
        source, split, _ = source_by_episode[ep]
        if (mapping.get('frame') != index or mapping.get('episode_index') != ep
                or mapping.get('step') != int(_scalar(row['frame_index']))
                or mapping.get('source_index') != source or mapping.get('split') != split):
            raise ValueError('Frame provenance and dataset disagree')
        grid, target_time = mapping.get('grid_time_monotonic'), mapping.get('target_time_monotonic')
        observation = mapping.get('observation', {})
        if (not isinstance(grid, (int, float)) or not isinstance(target_time, (int, float))
                or not math.isfinite(grid) or not math.isfinite(target_time)
                or (schema == 'fr5-stream-retarget-export-v1' and abs(target_time - grid - 1/FPS) > 1e-5)
                or target_time <= grid
                or observation.get('camera_time_monotonic', math.inf) > grid + 1e-8
                or observation.get('robot_time_monotonic', math.inf) > grid + 1e-8):
            raise ValueError('Observation and future target are not causally aligned')
        if not np.array_equal(np.asarray(row['action'], dtype=np.float32),
                              np.asarray(mapping.get('derived_action_radians_fraction'), dtype=np.float32)):
            raise ValueError('Frame action does not match derived source action')
        row['_physical_source'], row['_split'] = str(source), split
        row['_grid_time'] = float(grid)
        row['_camera_time'] = float(observation['camera_time_monotonic'])
        row['_source_capture_frame'] = observation['frame_index']
        row['_gripper_target_ref'] = mapping.get('gripper_target')
    episodes = validate_rows(rows)
    attach_observation_history(rows, episodes)
    attach_physical_observation_history(rows, episodes, proof)
    if set(episodes) != set(source_by_episode) or any(
            len(indices) != source_by_episode[ep][2] for ep, indices in episodes.items()):
        raise ValueError('Episode lengths differ from source provenance')
    train_eps, val_eps, _ = split_sources(rows, episodes)
    if (list(train_eps) != proof.get('train_episode_indices')
            or list(val_eps) != proof.get('validation_episode_indices')):
        raise ValueError('Split does not match exported provenance')
    exported_stats = json.loads((meta / 'stream_stats.json').read_text())
    if (exported_stats.get('episode_indices') != list(train_eps)
            or exported_stats.get('source_indices') != [i for i, s in enumerate(sources) if s['split'] == 'train']):
        raise ValueError('Statistics include the wrong sources or episodes')
    train_indices = [i for indices in train_eps.values() for i in indices]
    if exported_stats.get('frame_count') != len(train_indices):
        raise ValueError('Statistics have the wrong training frame count')
    for feature in ('observation.state', 'action'):
        data = np.asarray([rows[i][feature] for i in train_indices], dtype=np.float64)
        for name, q in (('q01', .01), ('q99', .99)):
            actual = np.asarray(exported_stats['stats'][feature][name])
            if not np.allclose(actual, np.quantile(data, q, axis=0), rtol=1e-6, atol=1e-7):
                raise ValueError(f'Exported {feature} statistics are not train-only')
    options = {'repo_id': 'local/fr5_pi05_stream', 'root': root, 'video_backend': 'pyav'}
    if 'token' in inspect.signature(LeRobotDataset).parameters:
        options['token'] = False
    dataset = LeRobotDataset(**options)
    if len(dataset) != len(rows):
        raise ValueError('LeRobot media index and rows differ')
    files = [meta / x for x in ('info.json', 'stats.json', 'tasks.parquet', 'stream_provenance.json',
                                'stream_source_frames.jsonl', 'stream_stats.json', 'stream_verification.json')]
    files += compact_files
    files += sorted((meta / 'episodes').rglob('*.parquet')) + sorted((meta / 'user_reviews').glob('*.json'))
    files += paths + sorted((root / 'videos').rglob('*.mp4'))
    hashes = {p.relative_to(root).as_posix(): checkpoints.digest(p) for p in files}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return rows, episodes, dataset, proof, fingerprint


def batch_for(indices, rows, episodes, dataset, pre, *, labels, action_representation=ABSOLUTE_REPRESENTATION,
              valid_device='cuda', image_profile=FULL_IMAGE_PROFILE,
              observation_context_profile=observation_history.NONE_PROFILE, cup_boundary_steps=0):
    import torch
    offsets = {i: (episode, n) for episode in episodes.values() for n, i in enumerate(episode)}
    samples = [dataset[i] for i in indices]
    cup_relative = observation_context_profile in observation_history.CUP_TCP_HISTORY_PROFILES
    state_key = '_cup_relative_state' if cup_relative else 'observation.state'
    history_key = ('_cup_relative_history' if cup_relative else '_physical_observation_history'
                   if observation_context_profile in observation_history.PHYSICAL_HISTORY_PROFILES else '_observation_history')
    value = {'observation.state': torch.tensor([rows[i][state_key] for i in indices], dtype=torch.float32),
             IMAGE_KEY: torch.stack([s[IMAGE_KEY] for s in samples]),
             'task': [observation_history.render_task(s['task'], rows[i].get(history_key),
                         observation_context_profile, initial_cup_uv=rows[i].get('_initial_cup_uv'),
                         cup_tracking=rows[i].get('_cup_tracking'), cup_timing=rows[i].get('_cup_timing'))
                      for i, s in zip(indices, samples)]}
    valid = None
    if labels:
        if cup_relative != (action_representation in checkpoints.CUP_REPRESENTATIONS):
            raise ValueError('Cup-relative action labels require measured cup-relative TCP input')
        action_key = ('_camera_aligned_actions' if observation_context_profile in observation_history.CUP_CAMERA_CLOCK_PROFILES
                      else '_cup_relative_action' if cup_relative else 'action')
        chunks = [chunk_for_anchor(rows, *offsets[i], action_key=action_key) for i in indices]
        absolute = torch.tensor([c[0] for c in chunks], dtype=torch.float32)
        value['action'] = absolute
        if action_representation in checkpoints.RELATIVE_REPRESENTATIONS:
            value['action'] = load_relative_helper().to_relative_actions(
                value['action'], value['observation.state'],
                input_representation=checkpoints.CUP_REPRESENTATION if cup_relative else ABSOLUTE_REPRESENTATION,
                per_tick=action_representation == checkpoints.CUP_TICK_DELTA_REPRESENTATION)
        elif action_representation not in (ABSOLUTE_REPRESENTATION, checkpoints.CUP_REPRESENTATION):
            raise ValueError('Unknown action representation')
        valid = torch.tensor([c[1] for c in chunks], dtype=torch.bool, device=valid_device)
    cup_uvs = [rows[i].get('_initial_cup_uv') for i in indices] if image_profile == checkpoints.CUP_CENTERED_IMAGE_PROFILE else None
    batch = prompt_integrity.preprocess(pre, apply_image_profile(value, image_profile, initial_cup_uv=cup_uvs))
    if observation_context_profile in observation_history.CUP_ACTION_PROFILES:
        batch['observation.cup_action_clock'] = torch.tensor(
            [observation_history.action_clock(s['task'], rows[i]['_cup_timing'], cup_tracking=rows[i]['_cup_tracking'],
                                              profile=observation_context_profile) for i, s in zip(indices, samples)],
            dtype=torch.float32, device=batch['observation.state'].device)
        if observation_context_profile in (observation_history.CUP_SETDOWN_CLEAN_PROFILE,
                                           observation_history.CUP_SETDOWN_FIXED_PROFILE):
            batch['observation.fixed_cup_action_clock'] = True
    if cup_boundary_steps:
        if not labels or observation_context_profile not in observation_history.CUP_CAMERA_CLOCK_PROFILES:
            raise ValueError('Cup boundaries require camera-aligned action labels')
        # The training-only mask uses the untouched future labels on the camera clock.
        original = [rows[i] for i in indices]
        timing = [row['_cup_timing'] for row in original]
        normal = torch.tensor([row['_cup_table_normal'].tolist() for row in original])
        targets = absolute[:, :cup_boundary_steps]
        delta = ((targets[..., :3] - value['observation.state'][:, None, :3]) * normal[:, None]).sum(-1) * 1000.
        boundary = (delta, targets[..., 6], normal,
                    torch.tensor([t['support_gap_mm'] or 0. for t in timing]),
                    torch.tensor([t['phase'] == 'hover' for t in timing]),
                    torch.tensor([t['phase'] in ('returning', 'at_table_height') for t in timing]),
                    torch.tensor([row['_cup_tracking']['native_contact'] == 1 for row in original]),
                    (value['observation.state'][:, 7] == 1.) & (value['observation.state'][:, 6] >= .98),
                    torch.tensor([row['_recorded_hold_remaining_s'] or 0. for row in original]))
        batch['cup_boundary'] = tuple(t.to(batch['observation.state'].device) for t in boundary)
        if observation_context_profile in observation_history.CUP_SETDOWN_PROFILES:
            device = batch['observation.state'].device
            batch['cup_settled_wait_mask'] = torch.tensor([
                row['_settled_wait_mask'] + [False] * (CHUNK - len(row['_settled_wait_mask'])) for row in original],
                dtype=torch.bool, device=device)
            batch['cup_setdown_age'] = torch.tensor([t['setdown_age_s'] or 0. for t in timing], device=device)
            if '_before_setdown_mask' in original[0]:
                batch['cup_before_setdown'] = torch.tensor([
                    (row['_before_setdown_mask'] + [False] * cup_boundary_steps)[:cup_boundary_steps]
                    for row in original], dtype=torch.bool, device=device)
        if '_pickup_wait_mask' in original[0]:
            batch['cup_pickup_wait_mask'] = torch.tensor([
                row['_pickup_wait_mask'] + [False] * (CHUNK - len(row['_pickup_wait_mask'])) for row in original],
                dtype=torch.bool, device=batch['observation.state'].device)
    return batch, valid








def validation_metrics(records, *, cup_relative=False):
    import numpy as np
    if not records:
        raise ValueError('No validation observations')
    gripper, closing, joint, tcp_xyz = [], [], [], []
    for state, target, prediction in records:
        state, target, prediction = map(lambda x: np.asarray(x, dtype=float), (state, target, prediction))
        gripper.append(abs(prediction[6] - target[6]))
        if target[6] < state[6] - 1e-5:
            closing.append(abs(prediction[6] - target[6]))
        error = np.abs(prediction[:6] - target[:6])
        if cup_relative:
            tcp_xyz.append(float(error[:3].mean()*1000))
            joint.append(float(np.degrees(error[3:]).mean()))
        else:
            joint.append(float(np.degrees(error).mean()))
    pose_metrics = ({'next_tcp_xyz_mae_mm': float(np.mean(tcp_xyz)),
                     'next_tcp_rpy_mae_deg': float(np.mean(joint))} if cup_relative else
                    {'next_joint_delta_mae_deg': float(np.mean(joint))})
    return {'samples': len(records), 'gripper_target_mae': float(np.mean(gripper)),
            'closing_target_count': len(closing),
            'closing_gripper_target_mae': float(np.mean(closing)) if closing else None,
            **pose_metrics,
            'evaluation_kind': 'heldout_physical_source_offline_diagnostic_not_task_success'}


def evaluate(policy, pre, post, dataset, rows, episodes, limit, seed,
             action_representation=ABSOLUTE_REPRESENTATION, image_profile=FULL_IMAGE_PROFILE,
             observation_context_profile=observation_history.NONE_PROFILE):
    import numpy as np
    import torch
    all_indices = [i for ep in episodes.values() for i in ep]
    closing = [i for i in all_indices if rows[i]['observation.state'][7] == 1
               and rows[i]['action'][6] < rows[i]['observation.state'][6] - 1e-5]
    close_count = min(len(closing), max(1, limit // 4)) if closing else 0
    indices = [closing[int(x)] for x in np.linspace(0, len(closing)-1, close_count, dtype=int)] if close_count else []
    remaining = min(limit - len(indices), len(all_indices))
    indices += [all_indices[int(x)] for x in np.linspace(0, len(all_indices)-1, remaining, dtype=int)] if remaining else []
    indices = list(dict.fromkeys(indices))
    records = []
    cup_relative = action_representation in checkpoints.CUP_REPRESENTATIONS
    state_key = '_cup_relative_state' if cup_relative else 'observation.state'
    policy.eval()
    with torch.random.fork_rng(devices=[0]):
        for n, i in enumerate(indices):
            torch.manual_seed(seed + n)
            batch, _ = batch_for([i], rows, episodes, dataset, pre, labels=False, image_profile=image_profile,
                                 observation_context_profile=observation_context_profile)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                actions = post(predict_action_chunk(policy, batch).detach().clone()).cpu().float()
            if action_representation in checkpoints.RELATIVE_REPRESENTATIONS:
                actions = load_relative_helper().to_absolute_actions(actions,
                    torch.tensor([rows[i][state_key]], dtype=torch.float32),
                    input_representation=action_representation)
            prediction = actions[0, 0].tolist()
            target = (rows[i]['_camera_aligned_actions'][0]
                      if observation_context_profile in observation_history.CUP_CAMERA_CLOCK_PROFILES
                      else rows[i]['_cup_relative_action' if cup_relative else 'action'])
            records.append((rows[i][state_key], target, prediction))
    policy.train()
    return validation_metrics(records, cup_relative=cup_relative)


def evaluate_transitions(policy, pre, post, dataset, rows, episodes, seed, draws=2,
                         action_representation=ABSOLUTE_REPRESENTATION, image_profile=FULL_IMAGE_PROFILE,
                         observation_context_profile=observation_history.NONE_PROFILE):
    """Evaluate first action and next six against actual target transitions."""
    import numpy as np
    import torch
    events_by_kind = transition_events(rows, episodes)
    records = []
    policy.eval()
    with torch.random.fork_rng(devices=[0]):
        for kind, events in events_by_kind.items():
            for event_number, event in enumerate(events):
                indices = episodes[event['episode']]
                for where, offset in (('before', -1), ('at', 0)):
                    position = event['position'] + offset
                    index = indices[position]
                    batch, _ = batch_for([index], rows, episodes, dataset, pre, labels=False,
                                         image_profile=image_profile, observation_context_profile=observation_context_profile)
                    action_key = ('_camera_aligned_actions' if observation_context_profile in observation_history.CUP_CAMERA_CLOCK_PROFILES
                                  else 'action')
                    targets, valid = chunk_for_anchor(rows, indices, position, action_key=action_key)
                    count = min(6, sum(valid))
                    target = float(targets[0][6])
                    alternate = event['new'] if abs(target-event['old']) < abs(target-event['new']) else event['old']
                    for draw in range(draws):
                        torch.manual_seed(seed + event['index']*10 + (offset+1)*3 + draw)
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                            actions = post(predict_action_chunk(policy, batch).detach().clone()).cpu().float()
                        if action_representation == RELATIVE_REPRESENTATION:
                            actions = load_relative_helper().to_absolute_actions(actions,
                                torch.tensor([rows[index]['observation.state']], dtype=torch.float32),
                                input_representation=RELATIVE_REPRESENTATION)
                        actions = actions[0]
                        if tuple(actions.shape) != (CHUNK, ACTION_DIM) or not bool(torch.isfinite(actions).all()):
                            raise ValueError('Invalid transition diagnostic prediction')
                        predicted = float(actions[0, 6])
                        records.append({'kind': kind, 'where': where, 'source': event['source'],
                            'event_index': event['index'], 'anchor_index': index, 'draw': draw,
                            'first_target': target, 'first_prediction': predicted,
                            'first_abs_error': abs(predicted-target),
                            'first_closer_to_recorded_target': abs(predicted-target) < abs(predicted-alternate),
                            'next6_abs_error': float(np.mean([abs(float(actions[i, 6])-targets[i][6])
                                                            for i in range(count)])),
                            'first6_predictions': [float(actions[i, 6]) for i in range(count)],
                            'first6_targets': [float(targets[i][6]) for i in range(count)]})
    policy.train()
    summary = {}
    for kind in ('close', 'open'):
        summary[kind] = {}
        for where in ('before', 'at'):
            group = [r for r in records if r['kind'] == kind and r['where'] == where]
            summary[kind][where] = {'events': len({r['event_index'] for r in group}), 'samples': len(group),
                'first_gripper_mae': float(np.mean([r['first_abs_error'] for r in group])) if group else None,
                'next6_gripper_mae': float(np.mean([r['next6_abs_error'] for r in group])) if group else None,
                'first_closer_to_recorded_target': sum(r['first_closer_to_recorded_target'] for r in group)}
    return {'summary': summary, 'records': records,
            'meaning': 'Recorded target matching diagnostic; not physical task success.'}


def build_lr_schedule(config, optimizer, name, learning_rate, total_steps, *, parent_report=None):
    sealed = parent_report.get('lr_schedule', {'name': 'constant'}) if parent_report is not None else None
    if sealed is not None and sealed['name'] != name:
        raise ValueError('Resume requires the same learning-rate schedule')
    if name == 'constant':
        return None, {'name': 'constant'}
    if name != 'cosine':
        raise ValueError('Unknown learning-rate schedule')
    if sealed is not None:
        total_steps = sealed['num_training_steps']
    preset = config.get_scheduler_preset()
    preset.peak_lr = learning_rate
    if total_steps < preset.num_decay_steps:
        effective_warmup = int(preset.num_warmup_steps * (total_steps / preset.num_decay_steps))
        effective_decay = total_steps
    else:
        effective_warmup = preset.num_warmup_steps
        effective_decay = preset.num_decay_steps
    plan = {'name': 'cosine', 'num_training_steps': total_steps,
            'num_warmup_steps': preset.num_warmup_steps, 'num_decay_steps': preset.num_decay_steps,
            'effective_warmup_steps': effective_warmup, 'effective_decay_steps': effective_decay,
            'peak_lr': preset.peak_lr, 'decay_lr': preset.decay_lr}
    if sealed is not None and sealed != plan:
        raise ValueError('Resume requires the sealed cosine schedule parameters')
    return preset.build(optimizer, num_training_steps=total_steps), plan


def restore_optimizer_and_scheduler(optimizer, scheduler, saved):
    optimizer.load_state_dict(saved['optimizer'])
    if scheduler is not None:
        if saved.get('scheduler') is None:
            raise ValueError('Cosine resume requires the saved scheduler state')
        scheduler.load_state_dict(saved['scheduler'])


def prepare_resume_metrics(run_path, step):
    metrics = run_path / 'metrics.jsonl'
    file_present = metrics.exists()
    available = set()
    if file_present:
        history, tail, unparsed = [], [], []
        with metrics.open('rb') as log:
            while True:
                line = log.readline()
                if not line:
                    break
                try:
                    recorded_step = json.loads(line)['step']
                    if not isinstance(recorded_step, int):
                        raise ValueError('Invalid metrics step')
                except (ValueError, KeyError, TypeError):
                    (tail if tail else unparsed).extend((line, log.read()))
                    break
                if recorded_step > step:
                    tail.append(line)
                else:
                    history.append(line)
                    if recorded_step >= 1:
                        available.add(recorded_step)
        needs_newline = bool(history and not history[-1].endswith(b'\n'))
        if needs_newline:
            history[-1] += b'\n'
        if tail or unparsed or needs_newline:
            for suffix_kind, contents in (('unsaved-after-step', tail), ('unparsed-at-resume-step', unparsed)):
                if not contents:
                    continue
                sequence = 1
                while True:
                    archive = run_path / f'metrics.{suffix_kind}-{step:06d}-{sequence:03d}.jsonl'
                    temporary = run_path / f'.{archive.name}.tmp'
                    if not archive.exists() and not temporary.exists():
                        break
                    sequence += 1
                with temporary.open('xb') as saved_tail:
                    saved_tail.writelines(contents)
                    saved_tail.flush()
                    os.fsync(saved_tail.fileno())
                temporary.rename(archive)
                print(json.dumps({'status': 'metrics_tail_preserved', 'resumed_from_step': step,
                                  'path': str(archive)}), flush=True)
            sequence = 1
            while True:
                temporary = run_path / f'.metrics.resume-step-{step:06d}-{sequence:03d}.tmp'
                if not temporary.exists():
                    break
                sequence += 1
            with temporary.open('xb') as active:
                active.writelines(history)
                active.flush()
                os.fsync(active.fileno())
            temporary.replace(metrics)
    missing = []
    previous = 0
    for recorded_step in sorted(available):
        if recorded_step > previous + 1:
            missing.append([previous + 1, recorded_step - 1])
        previous = recorded_step
    if previous < step:
        missing.append([previous + 1, step])
    return {'active_metrics': str(metrics), 'resumed_from_step': step,
            'checkpoint_history': {'file_present': file_present,
                                   'available_step_range': [min(available), max(available)] if available else None,
                                   'available_step_count': len(available), 'missing_step_ranges': missing},
            'preserved_unsaved_tails': [str(path) for path in
                                        sorted(run_path.glob('metrics.unsaved-after-step-*.jsonl'))],
            'preserved_unparsed_tails': [str(path) for path in
                                         sorted(run_path.glob('metrics.unparsed-at-resume-step-*.jsonl'))]}


def save_checkpoint(root, step, policy, optimizer, pre, post, stats, report, sampler, seen, base, fingerprint,
                    *, scheduler=None):
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    final = root / f'step-{step:06d}'
    temp = root / f'.step-{step:06d}.tmp'
    if final.exists() or temp.exists():
        raise FileExistsError(f'Checkpoint already exists: {final}')
    temp.mkdir()
    policy.config.save_pretrained(temp)
    policy.peft_config['default'].save_pretrained(temp)
    pre.save_pretrained(temp)
    post.save_pretrained(temp)
    adapter = get_peft_model_state_dict(policy)
    if not adapter or any(value.dtype != torch.float32 for value in adapter.values()):
        raise ValueError('Expected FP32 adapter master weights')
    save_file({k: v.detach().cpu().contiguous() for k, v in adapter.items()}, str(temp / 'adapter_model.safetensors'))
    save_file(projection_state(policy), str(temp / 'projections.safetensors'))
    mode = report.get('adaptation_mode', LORA_ADAPTATION)
    checkpoints.write_json(temp / 'adaptation.json', adaptation_metadata(mode))
    checkpoints.write_json(temp / 'action_representation.json',
                           action_representation_metadata(report.get('action_representation', ABSOLUTE_REPRESENTATION)))
    checkpoints.write_json(temp / 'image_profile.json',
                           checkpoints.image_profile_metadata(report.get('image_profile', FULL_IMAGE_PROFILE)))
    if mode in FULL_EXPERT_MODES:
        full = expert_state(policy)
        if not full or any(value.dtype != torch.float32 for value in full.values()):
            raise ValueError('Expected a complete FP32 action-expert snapshot')
        save_file(full, str(temp / 'expert_model.safetensors'))
    torch.save({'steps_completed': step, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'python_rng': sampler.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                'seen_anchors': sorted(seen)}, temp / 'optimizer.pt')
    normalization = {feature: {k: v.cpu().tolist() for k, v in values.items()} for feature, values in stats.items()}
    checkpoints.write_json(temp / 'normalization.json', normalization)
    checkpoints.write_json(temp / 'training-report.json', {**report, 'steps_completed': step})
    contract, source = load_contract()
    result = checkpoints.create_bundle(temp, contract=contract.contract(), base_weights=base,
        base_revision=legacy.REVISION, base_sha256=legacy.EXPECTED_WEIGHTS_SHA256,
        tokenizer_source=legacy.tokenizer_path(), tokenizer_revision=legacy.TOKENIZER_REVISION,
        contract_source=source, dataset_fingerprint=fingerprint)
    temp.rename(final)
    print(json.dumps({'status': 'checkpoint_saved', 'checkpoint': str(final),
                      'checkpoint_sha256': result['checkpoint_sha256'], 'steps_completed': step}), flush=True)


def validate_parent(parent_report, proof, args, *, resume):
    """Check source split and distinguish a warm start from an exact resume."""
    if parent_report.get('rank') != args.rank:
        raise ValueError('Parent LoRA rank differs')
    parent_proof = parent_report['source_provenance']
    def identity(source):
        return (source['source_scope_sha256'], source['user_review_sha256'], source['split'])
    parent_sources, sources = parent_proof['sources'], proof['sources']
    parent_ids = [identity(source) for source in parent_sources]
    source_ids = [identity(source) for source in sources]
    added_sources = parent_ids != source_ids
    if added_sources:
        error = 'Parent and current dataset must use the same physical source split or append captured reviewed sources'
        if (resume or len(sources) <= len(parent_sources)
                or parent_proof.get('training_data_kind') != 'captured'
                or proof.get('training_data_kind') != 'captured'):
            raise ValueError(error)
        # The preparation CLI groups added train sources before added validation
        # sources, so a later train source can precede a previous holdout.
        remaining = iter(sources)
        for old in parent_sources:
            current = next((source for source in remaining if identity(source) == identity(old)), None)
            if current is None or any(key in old and current.get(key) != old[key]
                                      for key in ('session', 'source_stream_sha256')):
                raise ValueError(error)
        # These identities are already emitted and checked by the exporter.
        # In particular, a held-out recording cannot be added again as train.
        for key in ('source_scope_sha256', 'user_review_sha256', 'session', 'source_stream_sha256'):
            if any(key in source for source in sources):
                values = [source.get(key) for source in sources]
                if None in values or len(set(values)) != len(values):
                    raise ValueError(error)
    if resume:
        resolve_observation_context_profile(getattr(args, 'observation_context_profile', None),
            parent_profile=parent_report.get('observation_context_profile', observation_history.NONE_PROFILE),
            resume=True)
        if parent_report.get('adaptation_mode', LORA_ADAPTATION) != args.adaptation:
            raise ValueError('Resume requires the same adaptation mode')
        if parent_report.get('action_representation', ABSOLUTE_REPRESENTATION) != args.action_representation:
            raise ValueError('Resume requires the same action representation')
        resolve_sampling_profile(getattr(args, 'sampling_profile', None),
                                 parent_report=parent_report, resume=True)
        for key in ('batch_size', 'gradient_accumulation', 'learning_rate', 'seed',
                    'first_action_weight', 'gripper_weight'):
            if parent_report.get(key) != getattr(args, key):
                raise ValueError('Resume training and loss parameters must match')
        if parent_report.get('executed_prefix_weight', 1.) != getattr(args, 'executed_prefix_weight', 1.):
            raise ValueError('Resume execution-prefix weight differs')
        if parent_report.get('vertical_weight', 1.) != getattr(args, 'vertical_weight', 1.):
            raise ValueError('Resume vertical loss weight differs')
    return added_sources


def resolve_adaptation_transition(parent_mode, requested_mode, *, resume=False, initialize=False):
    """Only explicit initialization may add vision adapters to a full expert."""
    if parent_mode not in ADAPTATION_MODES or requested_mode not in ADAPTATION_MODES:
        raise ValueError('Unknown adaptation mode')
    if resume and requested_mode != parent_mode:
        raise ValueError('Resume requires the same adaptation mode')
    if parent_mode == VISION_EXPERT_ADAPTATION and requested_mode != parent_mode:
        raise ValueError('Cannot silently discard saved vision adapters')
    if requested_mode == VISION_EXPERT_ADAPTATION and parent_mode != requested_mode:
        if not initialize or resume or parent_mode != FULL_EXPERT_ADAPTATION:
            raise ValueError('Vision upgrade requires explicit --initialize from a full-expert checkpoint')
        return True
    if parent_mode == FULL_EXPERT_ADAPTATION and requested_mode != FULL_EXPERT_ADAPTATION:
        raise ValueError('Cannot initialize LoRA-only mode from a full-expert checkpoint')
    return False


def run(args):
    if args.n_action_steps is not None:
        checkpoints.configured_action_steps({'n_action_steps': args.n_action_steps}, CHUNK)
    legacy.require_runtime_versions()
    legacy.configure_environment()
    dataset_path = legacy.check_data4(args.dataset)
    run_path = legacy.check_data4(args.run)
    if (legacy.ROOT / 'checkpoints') not in run_path.parents:
        raise ValueError('--run must be under data4/fr5-learning/checkpoints')
    base = legacy.check_data4(args.base) / 'model.safetensors'
    if not base.is_file() or legacy.sha(base) != legacy.EXPECTED_WEIGHTS_SHA256:
        raise ValueError('Pinned base model missing or SHA256 differs')
    selected = legacy.gpu_status(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected['index'])
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict  # noqa: F401
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Selected GPU was not isolated')
    rows, episodes, dataset, proof, fingerprint = read_dataset(dataset_path)
    train_eps, val_eps, heldout = split_sources(rows, episodes)
    requested_context = getattr(args, 'observation_context_profile', None)
    parent = legacy.check_data4(args.resume or args.initialize) if args.resume or args.initialize else None
    parent_info = checkpoints.verify_bundle(parent) if parent else None
    context_profile = resolve_observation_context_profile(requested_context,
        parent_profile=parent_info['observation_context_profile'] if parent_info else None, resume=bool(args.resume))
    aligned = context_profile in observation_history.CUP_CAMERA_CLOCK_PROFILES
    setdown = context_profile in observation_history.CUP_SETDOWN_PROFILES
    clean_setdown = context_profile == observation_history.CUP_SETDOWN_CLEAN_PROFILE
    cup_relative = args.action_representation in checkpoints.CUP_REPRESENTATIONS
    anchored_actions = args.action_representation in checkpoints.RELATIVE_REPRESENTATIONS
    cup_data = None
    if cup_relative:
        from cup_relative_stream_data import attach_cup_relative
        cup_data = attach_cup_relative(rows, episodes, proof, dataset_path)
    aligned_targets = None
    if aligned:
        if args.action_representation != checkpoints.CUP_TICK_DELTA_REPRESENTATION:
            raise ValueError('Camera-clock targets require cup-relative per-tick deltas')
        from cup_relative_stream_data import attach_camera_aligned_actions
        aligned_targets = attach_camera_aligned_actions(rows, episodes, proof)
    exported_stats = json.loads((dataset_path / 'meta/stream_stats.json').read_text())['stats']
    stats, fallbacks = (
        calculate_stats(rows, train_eps, cup_relative=True, cup_delta=anchored_actions) if cup_relative else
        calculate_stats(rows, train_eps, exported_stats))
    relative_stats_report = None
    if anchored_actions:
        relative_stats, relative_stats_report = load_relative_helper().compute_relative_action_stats(rows, train_eps,
            chunk_size=CHUNK, representation=args.action_representation, camera_aligned=aligned)
        stats['action'] = relative_stats['action']
        fallbacks = ([item for item in fallbacks if item['feature'] == 'observation.state']
                     + [{'feature': 'action', **item} for item in relative_stats_report['fallbacks']])
    current_train_stats = stats
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    parent_report = None
    vision_upgrade = False
    changing_to_cup = False
    rebase_stats = False
    if parent:
        previous = parent_info
        if previous['manifest']['policy_contract'] != load_contract()[0].contract():
            raise ValueError('Parent checkpoint contract differs')
        if args.resume and previous['manifest']['dataset_fingerprint'] != fingerprint:
            raise ValueError('Resume dataset differs from checkpoint')
        parent_report = json.loads((parent / 'training-report.json').read_text())
        added_sources = validate_parent(parent_report, proof, args, resume=bool(args.resume))
        changing_to_cup = (cup_relative and bool(args.initialize)
                           and parent_info['action_representation'] != args.action_representation)
        if parent_info['action_representation'] != args.action_representation and not changing_to_cup:
            raise ValueError('Different action representations require base initialization and their own train statistics')
        rebase_stats = (added_sources or changing_to_cup or
                        aligned != (parent_info['observation_context_profile'] in
                                    observation_history.CUP_CAMERA_CLOCK_PROFILES))
        vision_upgrade = resolve_adaptation_transition(parent_info['adaptation_mode'], args.adaptation,
                                                       resume=bool(args.resume), initialize=bool(args.initialize))
        n_action_steps = resolve_action_steps(args.n_action_steps, parent_steps=parent_info['n_action_steps'],
                                              resume=bool(args.resume))
        image_profile = resolve_image_profile(args.image_profile, parent_profile=parent_info['image_profile'],
                                              resume=bool(args.resume))
        saved_stats = json.loads((parent / 'normalization.json').read_text())
        if not rebase_stats:
            stats = {name: {key: torch.tensor(value, dtype=torch.float32) for key, value in values.items()}
                     for name, values in saved_stats.items()}
        for name, size in (('observation.state', STATE_DIM), ('action', ACTION_DIM)):
            if (set(stats.get(name, {})) != {'q01', 'q99'}
                    or stats[name]['q01'].shape != (size,) or stats[name]['q99'].shape != (size,)
                    or not torch.isfinite(stats[name]['q01']).all() or not torch.isfinite(stats[name]['q99']).all()
                    or not torch.all(stats[name]['q99'] > stats[name]['q01'])):
                raise ValueError('Parent checkpoint normalization is invalid')
        config = PI05Config.from_pretrained(parent, local_files_only=True, cli_overrides=['--device=cpu'])
        configure_observation_context(config, context_profile)
        if checkpoints.configured_action_steps({'n_action_steps': config.n_action_steps}, CHUNK) != parent_info['n_action_steps']:
            raise ValueError('Loaded policy n_action_steps differs from the sealed parent configuration')
        # A warm start may explicitly choose a new prefix; resume must preserve it.
        config.n_action_steps = n_action_steps
        if vision_upgrade:
            config.freeze_vision_encoder = False
        policy = load_policy(config, base, checkpoint=parent, adaptation=args.adaptation,
                             allow_vision_upgrade=vision_upgrade)
    else:
        image_profile = resolve_image_profile(args.image_profile)
        config = make_config(resolve_action_steps(args.n_action_steps), adaptation=args.adaptation)
        configure_observation_context(config, context_profile)
        policy = load_policy(config, base, rank=args.rank, adaptation=args.adaptation)
    requested_roll_span = getattr(args, 'cup_roll_state_min_span_rad', None)
    stats, roll_span, roll_training_range, roll_span_applied = configure_cup_roll_state_stats(
        stats, rows, train_eps, requested_roll_span, parent_report=parent_report, resume=bool(args.resume),
        cup_relative=cup_relative, rebase_stats=rebase_stats)
    cup_observations = (attach_initial_cup(rows, episodes, proof)
                        if context_profile in observation_history.CUP_HISTORY_PROFILES else None)
    if cup_relative != (context_profile in observation_history.CUP_TCP_HISTORY_PROFILES):
        raise ValueError('Cup-relative training requires matching TCP history and action representation')
    if image_profile == checkpoints.CUP_CENTERED_IMAGE_PROFILE and context_profile not in observation_history.CUP_HISTORY_PROFILES:
        raise ValueError('Cup-centered images require the initial cup observation')
    if args.resume and cup_data != parent_report.get('cup_relative_data'):
        raise ValueError('Resume cup-relative source provenance differs from the checkpoint')
    if args.resume and cup_observations != parent_report.get('initial_cup_observations'):
        raise ValueError('Resume initial cup observations differ from the checkpoint')
    sampling_profile = resolve_sampling_profile(getattr(args, 'sampling_profile', None),
                                               parent_report=parent_report, resume=bool(args.resume))
    cup_tracking_data = None
    if context_profile in observation_history.CUP_TRACKING_PROFILES:
        from cup_relative_stream_data import attach_cup_tracking
        cup_tracking_data = attach_cup_tracking(rows, proof, with_timing=context_profile in observation_history.CUP_TIMING_PROFILES,
                                               aligned=aligned, setdown=setdown, execution_steps=config.n_action_steps,
                                               mask_settled_wait_in_prefix=clean_setdown,
                                               mask_pickup_wait=setdown and sampling_profile == CUP_COUPLED_SAMPLING_PROFILE)
        if args.resume and cup_tracking_data != parent_report.get('cup_tracking_data'):
            raise ValueError('Resume tracking observations differ from the checkpoint')
    parameters = trainable_parameters(policy, args.adaptation)
    prefix_weight = getattr(args, 'executed_prefix_weight', 1.)
    vertical_weight = getattr(args, 'vertical_weight', 1.)
    if vertical_weight != 1. and not cup_relative:
        raise ValueError('Vertical loss weight requires TCP actions')
    prefix_loss = {'prefix_steps': config.n_action_steps, 'prefix_weight': prefix_weight, 'vertical_weight': vertical_weight}
    legacy.gpu_status(selected['index'])
    policy.to('cuda').train()
    config.device = 'cuda'
    pre, post = legacy.make_processors(config, stats, legacy.tokenizer_path())
    if anchored_actions:
        load_relative_helper().assert_manual_pipeline(config, preprocessor=pre, postprocessor=post)
    decay_learning_rate = getattr(args, 'decay_learning_rate', None)
    if decay_learning_rate is not None:
        config.scheduler_decay_lr = decay_learning_rate
    config.optimizer_lr = args.learning_rate
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(.9, .95), weight_decay=.01)
    initial = 0
    seen = set()
    sampler = make_sampler(sampling_profile, rows, train_eps, proof, args.seed)
    coupled_support = sampling_profile == CUP_COUPLED_SAMPLING_PROFILE
    support_focus = coupled_support or sampling_profile == CUP_SUPPORT_SAMPLING_PROFILE
    boundary_loss = support_focus or sampling_profile == CUP_BOUNDARY_SAMPLING_PROFILE
    xyz_loss_mode = resolve_xyz_loss_mode(getattr(args, 'xyz_loss_mode', None),
                                          parent_report=parent_report, resume=bool(args.resume))
    if xyz_loss_mode == 'terminal' and (not coupled_support or
            context_profile != observation_history.CUP_SETDOWN_FIXED_PROFILE):
        raise ValueError('Terminal XYZ training requires uncached fixed cup conditioning and coupled support')
    if setdown and (not coupled_support or proof['schema'] != 'fr5-stream-retarget-export-v1'):
        raise ValueError('Setdown supervision requires the full export and coupled support sampling')
    if boundary_loss:
        if (not aligned
                or args.action_representation != checkpoints.CUP_TICK_DELTA_REPRESENTATION
                or config.n_action_steps not in ((3, 10) if setdown else (10,))):
            raise ValueError('Camera-clock boundary training requires aligned cup tick deltas and the supported prefix')
        cup_action_scale = ((stats['action']['q99'] - stats['action']['q01']) / 2.).to('cuda')
    else:
        cup_action_scale = None
    saved = None
    if args.resume:
        if parent.parent != run_path:
            raise ValueError('Resume checkpoint must belong to the specified run')
        saved = torch.load(parent / 'optimizer.pt', map_location='cpu', weights_only=False)
        initial = int(saved['steps_completed'])
        if initial != parent_report['steps_completed'] or args.steps <= initial:
            raise ValueError('--steps is the total target and must exceed the resumed step')
    scheduler, lr_schedule = build_lr_schedule(config, optimizer, args.lr_schedule, args.learning_rate, args.steps,
                                               parent_report=parent_report if args.resume else None)
    if args.resume:
        restore_optimizer_and_scheduler(optimizer, scheduler, saved)
        sampler.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        seen = set(saved['seen_anchors'])
    else:
        run_path.mkdir(parents=True, exist_ok=False)
    if args.resume and not run_path.is_dir():
        raise ValueError('Resume --run directory must already exist')
    stats_difference = {name: {key: (current_train_stats[name][key] - stats[name][key]).tolist()
                               for key in ('q01', 'q99')}
                        for name in ('observation.state', 'action')}
    original_initialization = (parent_report if args.resume else {}) or {}
    terminal_report = None
    if proof['schema'] == 'fr5-stream-retarget-compact-v2':
        terminal_data = json.loads((dataset_path / proof['terminal_observations_file']).read_text())
        terminal_report = {'sources': len(terminal_data['sources']),
                           'raw_tail_observations': sum(len(x['tail_observations']) for x in terminal_data['sources']),
                           'used_as_training_anchors': False, 'future_action_labels_fabricated': False}
    used_fallbacks = copy.deepcopy(parent_report.get('normalization_fallbacks', []) if parent and not rebase_stats else fallbacks)
    if roll_span_applied:
        used_fallbacks = [item for item in used_fallbacks
                          if not (item.get('feature') == 'observation.state' and item.get('index') == 3)]
        used_fallbacks.append({'feature': 'observation.state', 'index': 3,
                               'reason': 'cup_roll_train_source_minimum_span', **roll_training_range})
    normalization_from = ('parent_checkpoint_unchanged' if parent and not rebase_stats else
                          'cup_frame_train_source_state_and_all_valid_tcp_delta_chunks'
                          if cup_relative and anchored_actions else
                          'cup_relative_train_sources_only' if cup_relative else
                          'train_source_state_and_all_valid_relative_chunks' if relative_stats_report else
                          'exported_training_sources_only_with_near_constant_fallbacks')
    if roll_span_applied:
        normalization_from = ('parent_checkpoint_with_train_roll_min_span' if parent else
                              normalization_from + '_with_train_roll_min_span')
    report = {'status': 'training_not_task_verified', 'dataset_path': str(dataset_path),
              'dataset_fingerprint': fingerprint, 'source_provenance': proof,
              'training_data_kind': proof['training_data_kind'],
              'terminal_observations': terminal_report,
              'heldout_physical_source': heldout, 'train_episodes': list(train_eps),
              'heldout_episodes': list(val_eps),
              'normalization_from': normalization_from,
              'normalization_fallbacks': used_fallbacks,
              'dataset_suggested_normalization_fallbacks': fallbacks,
              'current_train_stats_minus_used_stats': stats_difference,
              'normalization_checkpoint_sha256': (parent_info['checkpoint_sha256'] if parent and not rebase_stats else None),
              'cup_roll_state_min_span_requested_rad': requested_roll_span,
              'cup_roll_state_min_span_rad': roll_span,
              'cup_roll_state_bounds_rad': ([float(stats['observation.state'][key][3]) for key in ('q01', 'q99')]
                                            if cup_relative else None),
              'cup_roll_state_training_range_rad': roll_training_range,
              'camera_aligned_targets': aligned_targets,
              'action_label_origin': 'derived_from_measured_trajectory',
              'gripper_target_origin': 'recorded_gripper_command_or_explicit_non_rpc_initial_hold',
              'target_timebase_hz': FPS, 'chunk_size': CHUNK, 'n_action_steps': config.n_action_steps,
              'parent_n_action_steps': parent_info['n_action_steps'] if parent_info else None,
              'tail_attention_and_loss_mask': True,
              'sampling': sampler.VERSION,
              'sampling_profile': sampling_profile,
              'adaptation_mode': args.adaptation,
              'action_representation': args.action_representation,
              'image_profile': image_profile,
              'observation_context_profile': context_profile,
              'initial_cup_observations': cup_observations,
              'cup_relative_data': cup_data,
              'cup_tracking_data': cup_tracking_data,
              'prefix_training': 'uncached',
              'relative_action_statistics': relative_stats_report,
              'adaptation': adaptation_metadata(args.adaptation),
              'vision_lora': vision_adapter_report(policy, args.adaptation),
              'vision_lora_initialization': (None if args.adaptation != VISION_EXPERT_ADAPTATION else
                   'explicit_full_expert_initialize_zero_b' if vision_upgrade else
                   'loaded_from_checkpoint' if parent else 'fresh_base_zero_b'),
              'model_architecture': 'official_pi05_with_peft_adapters',
              'trainable_expert_parameters': sum(p.numel() for name, p in policy.named_parameters()
                                                 if is_expert_base(name) and p.requires_grad),
              'rank': args.rank, 'batch_size': args.batch_size,
              'gradient_accumulation': args.gradient_accumulation,
              'effective_batch_size': args.batch_size * args.gradient_accumulation,
              'learning_rate': args.learning_rate,
              'lr_schedule': lr_schedule, 'target_steps': args.steps,
              'first_action_weight': args.first_action_weight, 'gripper_weight': args.gripper_weight,
              'executed_prefix_weight': prefix_weight,
              'vertical_weight': vertical_weight,
              'xyz_loss_mode': xyz_loss_mode,
              'xyz_terminal_steps': config.num_inference_steps if xyz_loss_mode == 'terminal' else None,
              'xyz_terminal_coefficient': 6. if xyz_loss_mode == 'terminal' else None,
              'xyz_terminal_margin_mm': .25 if xyz_loss_mode == 'terminal' else None,
              'cup_boundary_loss': ('stable_setdown_active_prefix_pickup_wait_mask_and_terminal_xyz_prefix_v1' if xyz_loss_mode == 'terminal' else
                                    'stable_setdown_fixed_observation_pickup_wait_mask_and_xyz_prefix_v6' if clean_setdown else
                                    'stable_setdown_active_prefix_pickup_wait_mask_and_xyz_prefix_v6' if setdown else
                                    'signed_clean_action_error_coupled_support_xyz_prefix_v6' if coupled_support else
                                    'signed_clean_action_error_raw_hold_and_support_v1' if support_focus else
                                    'signed_clean_action_error_mm_and_percentage_points_v1' if boundary_loss else None),
              'trainable_dtype': 'float32', 'frozen_base_dtype': 'bfloat16_with_upstream_fp32_vision_and_norms',
              'trainable_parameters': sum(p.numel() for p in parameters), 'gpu_index': selected['index'],
              'seed': args.seed, 'resumed_from': str(parent) if args.resume else None,
              'initialized_from': str(parent) if args.initialize else original_initialization.get('initialized_from'),
              'initialized_from_sha256': (parent_info['checkpoint_sha256'] if args.initialize else
                                           original_initialization.get('initialized_from_sha256')),
              'initialized_from_steps': (parent_report['steps_completed'] if args.initialize else
                                         original_initialization.get('initialized_from_steps')),
              'optimizer_restored': bool(args.resume),
              'optimizer_migration': ('restored_same_sampler' if args.resume else
                                      'fresh_adamw_for_new_sampling_distribution'),
              'sampler_migration': ('restored_same_sampler' if args.resume else 'fresh_seed_for_new_sampler'),
              'robot_commands_sent': 0}
    if xyz_loss_mode == 'terminal':
        from stream_terminal_loss import move_optimizer_moments
        report['terminal_memory_method'] = 'checkpoint_native_denoise_steps_cpu_adamw_moments_between_gpu_updates'
    started = time.monotonic()
    remaining = {index: (len(rows[index]['_camera_aligned_actions']) if aligned else min(CHUNK, len(indices)-offset))
                 for indices in train_eps.values() for offset, index in enumerate(indices)}
    if args.resume:
        report['metrics_lineage'] = prepare_resume_metrics(run_path, initial)
    with (run_path / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        for step in range(initial + 1, args.steps + 1):
            samples = sampler.draw_balanced(args.batch_size * args.gradient_accumulation)
            indices = [index for index, _ in samples]
            sampling_details = [detail for _, detail in samples]
            seen.update(indices)
            optimizer.zero_grad(set_to_none=True)
            if xyz_loss_mode == 'terminal':
                move_optimizer_moments(optimizer, offload=True)
            microbatches = [indices[i:i+args.batch_size] for i in range(0, len(indices), args.batch_size)]
            weights = [loss_normalizer([remaining[i] for i in group], args.first_action_weight, args.gripper_weight,
                                       **prefix_loss)
                       for group in microbatches]
            total_weight = sum(weights)
            loss_value = 0.
            loss_components = defaultdict(float)
            for group, weight in zip(microbatches, weights):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    parts = {} if boundary_loss else None
                    batch, valid = batch_for(group, rows, train_eps, dataset, pre, labels=True,
                                             action_representation=args.action_representation,
                                             image_profile=image_profile,
                                             observation_context_profile=context_profile,
                                             cup_boundary_steps=config.n_action_steps if boundary_loss else 0)
                    loss = masked_flow_loss(policy, batch, valid, first_action_weight=args.first_action_weight,
                                            gripper_weight=args.gripper_weight, cup_action_scale=cup_action_scale,
                                            support_focus=support_focus, coupled_support=coupled_support,
                                            xyz_loss_mode=xyz_loss_mode, loss_parts=parts,
                                            backward_scale=weight / total_weight if xyz_loss_mode == 'terminal' else None,
                                            **prefix_loss)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                (loss * (weight / total_weight)).backward()
                loss_value += float(loss.detach()) * weight / total_weight
                if parts is not None:
                    for name, value in parts.items():
                        loss_components[name] += value * weight / total_weight
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError('Nonfinite or absent gradient')
            if step == initial + 1:
                grad_dtypes = sorted({str(p.grad.dtype) for p in parameters if p.grad is not None})
                if grad_dtypes != ['torch.float32']:
                    raise RuntimeError(f'Expected FP32 training gradients, got {grad_dtypes}')
                components = gradient_summary(policy)
                if (args.adaptation in FULL_EXPERT_MODES
                        and ('action_expert' not in components or components['action_expert']['norm'] <= 0
                             or 'language_lora' not in components or components['language_lora']['norm'] <= 0
                             or 'expert_lora' in components)):
                    raise RuntimeError('Full-expert or language gradients are missing or expert LoRA is unfrozen')
                if (args.adaptation == VISION_EXPERT_ADAPTATION
                        and ('vision_lora' not in components or components['vision_lora']['norm'] <= 0)):
                    raise RuntimeError('Vision LoRA gradients are missing')
                print(json.dumps({'status': 'first_gradient', 'dtypes': grad_dtypes,
                                  'parameters_with_gradient': sum(p.grad is not None for p in parameters),
                                  'effective_batch_size': len(indices),
                                  'components': components,
                                  'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated()/1024**2, 2)}), flush=True)
            step_learning_rate = optimizer.param_groups[0]['lr']
            if xyz_loss_mode == 'terminal':
                move_optimizer_moments(optimizer, offload=False)
            optimizer.step()
            if xyz_loss_mode == 'terminal':
                moment_report = move_optimizer_moments(optimizer, offload=True)
            if scheduler is not None:
                scheduler.step()
            if step == initial + 1:
                optimizer_dtypes = sorted({str(state[key].dtype) for state in optimizer.state.values()
                                           for key in ('exp_avg', 'exp_avg_sq') if key in state})
                if optimizer_dtypes != ['torch.float32']:
                    raise RuntimeError(f'Expected FP32 AdamW states, got {optimizer_dtypes}')
                print(json.dumps({'status': 'first_optimizer_step', 'state_dtypes': optimizer_dtypes,
                                  'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated()/1024**2, 2)}), flush=True)
            item = {'step': step, 'loss': loss_value, 'gradient_norm': float(norm),
                    'learning_rate': step_learning_rate,
                    'elapsed_s': time.monotonic() - started, 'sampling': sampling_details}
            if loss_components:
                item['loss_components'] = dict(loss_components)
            if xyz_loss_mode == 'terminal':
                item['optimizer_moments'] = moment_report
                item['gpu_allocated_mib'] = round(torch.cuda.memory_allocated()/1024**2, 2)
                item['gpu_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated()/1024**2, 2)
            log.write(json.dumps(item) + '\n')
            log.flush()
            if step == initial + 1 or step % 25 == 0:
                print(json.dumps({'status': 'training', **item}), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                report['heldout_diagnostics'] = evaluate(policy, pre, post, dataset, rows, val_eps,
                                                         args.eval_samples, args.seed, args.action_representation,
                                                         image_profile, context_profile)
                report['transition_diagnostics'] = {
                    'train': evaluate_transitions(policy, pre, post, dataset, rows, train_eps,
                                                  args.seed, args.transition_eval_draws, args.action_representation,
                                                  image_profile, context_profile),
                    'validation': evaluate_transitions(policy, pre, post, dataset, rows, val_eps,
                                                       args.seed, args.transition_eval_draws, args.action_representation,
                                                       image_profile, context_profile)}
                report['last_loss'] = item['loss']
                report['anchor_coverage'] = anchor_coverage(rows, train_eps, seen)
                report['sampler_report'] = sampler.report()
                save_checkpoint(run_path, step, policy, optimizer, pre, post, stats, report, sampler, seen, base,
                                fingerprint, scheduler=scheduler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True, help='Total optimizer steps including a resumed checkpoint')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--adaptation', choices=ADAPTATION_MODES, default=LORA_ADAPTATION)
    parser.add_argument('--action-representation', choices=(ABSOLUTE_REPRESENTATION, RELATIVE_REPRESENTATION,
                                                            *checkpoints.CUP_REPRESENTATIONS),
                        default=ABSOLUTE_REPRESENTATION)
    parser.add_argument('--n-action-steps', type=int, default=None,
                        help='Leading actions before another observation (1..50; default 50 or inherit parent)')
    parser.add_argument('--image-profile', choices=checkpoints.IMAGE_PROFILES, default=None,
                        help='Camera views (default full-v1; resume inherits the sealed parent)')
    parser.add_argument('--sampling-profile', choices=SAMPLING_PROFILES, default=None,
                        help='Original balanced anchors or extra recorded open-approach anchors')
    parser.add_argument('--observation-context-profile', choices=observation_history.PROFILES, default=None,
                        help='Observation history text profile (default none or inherit parent)')
    parser.add_argument('--cup-roll-state-min-span-rad', type=float, default=None,
                        help='Minimum cup-relative state roll range; omitted values inherit the parent')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Microbatches per update; effective batch must be divisible by four')
    parser.add_argument('--base', type=Path, default=legacy.ROOT / 'models/pi05_base')
    parents = parser.add_mutually_exclusive_group()
    parents.add_argument('--resume', type=Path, help='Restore this run including optimizer and sampler state')
    parents.add_argument('--initialize', type=Path,
                         help='Warm-start weights; use current train statistics when sources or representation change')
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--decay-learning-rate', type=float, default=None,
                        help='Final cosine learning rate (default from the policy configuration)')
    parser.add_argument('--lr-schedule', choices=('constant', 'cosine'), default='constant',
                        help='Learning-rate schedule (resume must match the checkpoint)')
    parser.add_argument('--first-action-weight', type=float, default=8.)
    parser.add_argument('--executed-prefix-weight', type=float, default=1.,
                        help='Loss weight on consumed points after the first; the tail stays weight one')
    parser.add_argument('--gripper-weight', type=float, default=3.)
    parser.add_argument('--vertical-weight', type=float, default=1., help='Weight for measured TCP Z targets')
    parser.add_argument('--xyz-loss-mode', choices=('clean', 'terminal'), default=None,
                        help='Supervise cup translation with a clean estimate or the complete Euler sample')
    parser.add_argument('--transition-eval-draws', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--eval-samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261102)
    args = parser.parse_args()
    if args.n_action_steps is None and not (args.resume or args.initialize):
        args.n_action_steps = CHUNK
    if args.image_profile is None and not (args.resume or args.initialize):
        args.image_profile = FULL_IMAGE_PROFILE
    if (min(args.steps, args.rank, args.batch_size, args.gradient_accumulation, args.save_every,
            args.eval_samples, args.transition_eval_draws) < 1
            or (args.n_action_steps is not None and not 1 <= args.n_action_steps <= CHUNK)
            or args.batch_size * args.gradient_accumulation % 4
            or not 0 < args.learning_rate <= .01
            or (args.decay_learning_rate is not None and not 0 < args.decay_learning_rate <= .01)
            or (args.cup_roll_state_min_span_rad is not None
                and (not math.isfinite(args.cup_roll_state_min_span_rad) or args.cup_roll_state_min_span_rad <= 0.))
            or not 0 < args.first_action_weight <= 100
            or not 0 < args.executed_prefix_weight <= 100
            or not 0 < args.vertical_weight <= 100
            or not 0 < args.gripper_weight <= 100):
        parser.error('Invalid training parameters')
    run(args)


if __name__ == '__main__':
    main()
