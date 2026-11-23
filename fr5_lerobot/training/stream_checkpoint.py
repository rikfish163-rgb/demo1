"""Immutable, independently inspectable checkpoints for the FR5 stream policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import stream_prompt_integrity as prompt_integrity
import stream_observation_history as observation_history

SCHEMA = 'fr5-stream-checkpoint-v1'
RELATIVE_SCHEMA = 'fr5-stream-relative-checkpoint-v1'
MANIFEST = 'stream-manifest.json'
ABSOLUTE_REPRESENTATION = 'absolute-joint-position-v1'
RELATIVE_REPRESENTATION = 'joint-delta-from-anchor-v1'
CUP_REPRESENTATION = 'cup-relative-tcp-v1'
CUP_DELTA_REPRESENTATION = 'cup-relative-tcp-delta-v1'
CUP_TICK_DELTA_REPRESENTATION = 'cup-relative-tcp-tick-delta-v1'
CUP_REPRESENTATIONS = (CUP_REPRESENTATION, CUP_DELTA_REPRESENTATION, CUP_TICK_DELTA_REPRESENTATION)
RELATIVE_REPRESENTATIONS = (RELATIVE_REPRESENTATION, CUP_DELTA_REPRESENTATION, CUP_TICK_DELTA_REPRESENTATION)
LORA_ADAPTATION = 'lora'
FULL_EXPERT_ADAPTATION = 'full-expert-language-lora'
VISION_EXPERT_ADAPTATION = 'full-expert-vision-language-lora'
FULL_EXPERT_MODES = (FULL_EXPERT_ADAPTATION, VISION_EXPERT_ADAPTATION)
ADAPTATION_MODES = (LORA_ADAPTATION, *FULL_EXPERT_MODES)
BASE_LORA_TARGETS = (r'.*\.(gemma_expert\.model|paligemma\.model\.language_model)'
                     r'\..*\.self_attn\.(q|v)_proj')
VISION_LORA_TARGETS = (r'.*\.paligemma\.model\.vision_tower\.vision_model\.encoder'
                       r'\.layers\.\d+\.self_attn\.(q|v)_proj')
VISION_EXPERT_LORA_TARGETS = rf'(?:{BASE_LORA_TARGETS}|{VISION_LORA_TARGETS})'
FULL_IMAGE_PROFILE = 'full-v1'
WORKSPACE_IMAGE_PROFILE = 'full-and-workspace-v1'
WORKSPACE_ONLY_IMAGE_PROFILE = 'workspace-only-v1'
CUP_CENTERED_IMAGE_PROFILE = 'cup-centered-v1'
IMAGE_PROFILES = (FULL_IMAGE_PROFILE, WORKSPACE_IMAGE_PROFILE, WORKSPACE_ONLY_IMAGE_PROFILE,
                  CUP_CENTERED_IMAGE_PROFILE)
REQUIRED = {
    'config.json', 'adapter_config.json', 'adapter_model.safetensors',
    'projections.safetensors', 'optimizer.pt', 'policy_preprocessor.json',
    'policy_postprocessor.json', 'normalization.json', 'training-report.json',
    'stream_contract.py', 'train_stream_pi05.py', 'serve_stream_pi05.py',
    'stream_checkpoint.py', 'train_pi05_fr5.py',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2, allow_nan=False) + '\n', encoding='utf-8')


def configured_action_steps(config, chunk_size=50):
    """Read the execution prefix, including checkpoints predating the field."""
    steps = config.get('n_action_steps', chunk_size)
    if type(steps) is not int or not 1 <= steps <= chunk_size:
        raise ValueError(f'n_action_steps must be an integer between 1 and {chunk_size}')
    return steps


def image_profile_metadata(name):
    """Describe both views explicitly; the second view is never a wrist camera."""
    if name not in IMAGE_PROFILES:
        raise ValueError('Unknown image profile')
    workspace = name != FULL_IMAGE_PROFILE
    only = name in (WORKSPACE_ONLY_IMAGE_PROFILE, CUP_CENTERED_IMAGE_PROFILE)
    info = {'schema': 'fr5-stream-image-profile-v1', 'name': name,
            'source_key': 'observation.images.front',
            'workspace_key': (('observation.images.front' if only else
                               'observation.images.left_wrist_0_rgb') if workspace else None),
            'workspace_crop_xyxy': ([240, 100, 640, 360] if workspace else None),
            'workspace_crop_coordinates': ('xyxy_half_open' if workspace else None),
            'workspace_view_source': ('crop_of_front_camera' if workspace else None),
            'workspace_is_physical_wrist_camera': False, 'physical_camera_count': 1}
    if only:
        info['full_view_retained'] = False
        info['output_image_keys'] = ['observation.images.front']
    if name == CUP_CENTERED_IMAGE_PROFILE:
        info.update(workspace_crop_xyxy=None, workspace_view_source='cup_centered_crop_of_front_camera',
                    crop_size_wh=[384, 256], cup_anchor_crop_xy=[230, 115],
                    cup_uv_source_image_size_wh=[1920, 1080], source_image_size_wh=[640, 360],
                    edge_padding='replicate')
    return info


def read_image_profile(root):
    """Check the declared profile; older checkpoints have a full view only."""
    root = Path(root)
    path = root / 'image_profile.json'
    report = json.loads((root / 'training-report.json').read_text(encoding='utf-8'))
    if not path.is_file():
        if report.get('image_profile', FULL_IMAGE_PROFILE) != FULL_IMAGE_PROFILE:
            raise ValueError('Checkpoint lacks image-profile metadata')
        return image_profile_metadata(FULL_IMAGE_PROFILE)
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value != image_profile_metadata(value.get('name')):
        raise ValueError('Invalid checkpoint image-profile metadata')
    if report.get('image_profile', FULL_IMAGE_PROFILE) != value['name']:
        raise ValueError('Checkpoint image profile disagrees with training report')
    return value


def validate_context_tokenizer(root, config, profile):
    if profile not in observation_history.HISTORY_PROFILES:
        return
    processor = json.loads((Path(root) / 'policy_preprocessor.json').read_text(encoding='utf-8'))
    tokenizers = [step for step in processor.get('steps', [])
                  if step.get('registry_name') == 'tokenizer_processor']
    length = prompt_integrity.MAX_PROMPT_TOKENS
    if (config.get('tokenizer_max_length') != length or len(tokenizers) != 1
            or tokenizers[0].get('config', {}).get('max_length') != length
            or tokenizers[0].get('config', {}).get('padding') != 'max_length'):
        raise ValueError('History checkpoint requires matching fixed 512-token configuration and preprocessor')


def safe_file(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe bundle path: {name}')
    full = (root / path).resolve()
    if root not in full.parents or not full.is_file():
        raise ValueError(f'Missing or unsafe bundle file: {name}')
    return full


def read_adaptation(root):
    """Old sealed checkpoints predate adaptation metadata and use LoRA."""
    root = Path(root)
    path = root / 'adaptation.json'
    if not path.is_file():
        report = json.loads((root / 'training-report.json').read_text(encoding='utf-8'))
        if report.get('adaptation_mode', LORA_ADAPTATION) != LORA_ADAPTATION:
            raise ValueError('Full-expert checkpoint lacks adaptation metadata')
        return {'schema': 'fr5-stream-adaptation-v1', 'mode': LORA_ADAPTATION, 'legacy_checkpoint': True}
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('schema') != 'fr5-stream-adaptation-v1'
            or value.get('mode') not in ADAPTATION_MODES):
        raise ValueError('Invalid checkpoint adaptation metadata')
    if value['mode'] == VISION_EXPERT_ADAPTATION:
        required = {'vision': 'siglip_encoder_q_v_lora', 'language': 'q_v_lora',
                    'action_projections': 'full_fp32', 'action_expert': 'full_model_fp32',
                    'inherited_expert_lora': 'frozen_and_saved'}
        if any(value.get(key) != wanted for key, wanted in required.items()):
            raise ValueError('Invalid vision-adaptation metadata')
    return value


def read_action_representation(root):
    """The model representation may differ from the absolute wire contract."""
    root = Path(root)
    path = root / 'action_representation.json'
    if not path.is_file():
        report = json.loads((root / 'training-report.json').read_text(encoding='utf-8'))
        if report.get('action_representation', ABSOLUTE_REPRESENTATION) != ABSOLUTE_REPRESENTATION:
            raise ValueError('Relative checkpoint lacks action-representation metadata')
        return {'schema': 'fr5-stream-action-representation-v1', 'mode': ABSOLUTE_REPRESENTATION,
                'legacy_checkpoint': True}
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('schema') != 'fr5-stream-action-representation-v1'
            or value.get('mode') not in (ABSOLUTE_REPRESENTATION, RELATIVE_REPRESENTATION, *CUP_REPRESENTATIONS)):
        raise ValueError('Invalid checkpoint action-representation metadata')
    if (value['mode'] == RELATIVE_REPRESENTATION
            and (value.get('wire_mode') != ABSOLUTE_REPRESENTATION
                 or value.get('joint_dimensions') != 6 or value.get('gripper_representation') != 'absolute'
                 or value.get('anchor') != 'raw_request_state'
                 or value.get('official_relative_processor_enabled') is not False)):
        raise ValueError('Relative checkpoint does not declare absolute wire and same-request anchoring')
    if value['mode'] in CUP_REPRESENTATIONS and value != cup_action_metadata(value['mode']):
        raise ValueError('Cup action representation differs from its declared pose and wire units')
    return value


def cup_action_metadata(mode=CUP_REPRESENTATION):
    if mode not in CUP_REPRESENTATIONS:
        raise ValueError('Unknown cup action representation')
    result = {'schema': 'fr5-stream-action-representation-v1', 'mode': mode,
            'wire_mode': CUP_REPRESENTATION, 'controller_mode': ABSOLUTE_REPRESENTATION,
            'dimensions': ['tcp_x_m_from_cup', 'tcp_y_m_from_cup', 'tcp_z_m_from_cup',
                           'tcp_rx_rad_in_base', 'tcp_ry_rad_in_base', 'tcp_rz_rad_in_base', 'gripper_fraction'],
            'anchor': 'initial_cup_base_in_J1', 'official_relative_processor_enabled': False}
    if mode in (CUP_DELTA_REPRESENTATION, CUP_TICK_DELTA_REPRESENTATION):
        result.update(dimensions=['delta_tcp_x_m', 'delta_tcp_y_m', 'delta_tcp_z_m',
                                  'delta_tcp_rx_rad', 'delta_tcp_ry_rad', 'delta_tcp_rz_rad', 'gripper_fraction'],
                      anchor='raw_request_cup_relative_state', cup_origin='initial_cup_base_in_J1',
                      gripper_representation='absolute')
    if mode == CUP_TICK_DELTA_REPRESENTATION:
        result['horizon_divisor'] = 'one_based_action_index'
    return result


def create_bundle(root, *, contract, base_weights, base_revision, base_sha256,
                  tokenizer_source, tokenizer_revision, contract_source, dataset_fingerprint):
    root = Path(root).resolve()
    if not root.is_dir() or (root / MANIFEST).exists():
        raise ValueError('Expected an unsealed checkpoint directory')
    base = Path(base_weights).resolve()
    if not base.is_file() or digest(base) != base_sha256:
        raise ValueError('Base model SHA256 mismatch')
    tokenizer = Path(tokenizer_source)
    if not tokenizer.is_dir() or tokenizer.name != tokenizer_revision:
        raise ValueError('Pinned tokenizer snapshot missing')
    target = root / 'tokenizer'
    target.mkdir()
    for f in sorted(tokenizer.iterdir()):
        if f.is_file():
            shutil.copyfile(f, target / f.name)
    source = Path(__file__).resolve().parent
    for name in ('train_stream_pi05.py', 'serve_stream_pi05.py', 'stream_checkpoint.py',
                 'stream_prompt_integrity.py', 'stream_conditioned_sampling.py', 'stream_observation_history.py',
                 'stream_terminal_loss.py', 'train_pi05_fr5.py'):
        shutil.copyfile(source / name, root / name)
    shutil.copyfile(contract_source, root / 'stream_contract.py')
    helpers = root / 'record_helpers'
    helpers.mkdir()
    for source_name, destination in (('lerobot_camera_gemini/timing.py', 'camera_timing.py'),
                                      ('lerobot_robot_fairino/feedback.py', 'gripper_feedback.py')):
        shutil.copyfile(source.parent / 'src' / source_name, helpers / destination)
    report = json.loads((root / 'training-report.json').read_text(encoding='utf-8'))
    context_profile = report.get('observation_context_profile', observation_history.NONE_PROFILE)
    context_metadata = observation_history.context_metadata(context_profile)
    context_path = root / 'observation_context.json'
    if not context_path.exists():
        write_json(context_path, context_metadata)
    elif json.loads(context_path.read_text(encoding='utf-8')) != context_metadata:
        raise ValueError('Observation context disagrees with training report')
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    validate_context_tokenizer(root, config, context_profile)
    if not (root / 'image_profile.json').exists():
        write_json(root / 'image_profile.json',
                   image_profile_metadata(report.get('image_profile', FULL_IMAGE_PROFILE)))
    image_profile = read_image_profile(root)
    shutil.copyfile(source / 'stream_image_profile.py', root / 'stream_image_profile.py')
    n_action_steps = configured_action_steps(config, contract['chunk_size'])
    if 'n_action_steps' in report and (type(report['n_action_steps']) is not int
                                        or report['n_action_steps'] != n_action_steps):
        raise ValueError('Checkpoint action execution steps disagree with training report')
    representation = read_action_representation(root)
    if representation['mode'] in RELATIVE_REPRESENTATIONS:
        shutil.copyfile(source / 'relative_stream_actions.py', root / 'relative_stream_actions.py')
    if representation['mode'] in CUP_REPRESENTATIONS:
        shutil.copyfile(source / 'cup_relative_stream_data.py', root / 'cup_relative_stream_data.py')
    if context_profile in observation_history.CUP_TRACKING_PROFILES:
        for name in ('cup_tracking.py', 'rgbd_geometry.py', 'cup_rim_geometry.py'):
            shutil.copyfile(source.parents[1] / 'fr5_jev' / name, root / name)
    if context_profile in observation_history.CUP_TIMING_PROFILES:
        shutil.copyfile(source.parents[1] / 'fr5_jev/cup_timing.py', root / 'cup_timing.py')
    if report.get('dataset_fingerprint') != dataset_fingerprint:
        raise ValueError('Training report dataset fingerprint mismatch')
    if any(not (root / name).is_file() for name in REQUIRED):
        raise ValueError('Checkpoint is missing a required file')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name != MANIFEST)
    files = {p.relative_to(root).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in paths}
    manifest = {'schema': RELATIVE_SCHEMA if representation['mode'] in RELATIVE_REPRESENTATIONS else SCHEMA,
                'policy_contract': contract,
                'base': {'weights_path': str(base), 'revision': base_revision,
                         'sha256': base_sha256, 'bytes': base.stat().st_size},
                'tokenizer_revision': tokenizer_revision, 'dataset_fingerprint': dataset_fingerprint,
                'steps_completed': report['steps_completed'],
                'n_action_steps': n_action_steps,
                'image_profile': image_profile['name'],
                'observation_context_profile': context_profile,
                'action_representation': representation['mode'],
                'prompt_integrity': dict(prompt_integrity.PROMPT_INTEGRITY), 'files': files}
    adaptation = read_adaptation(root)
    if adaptation['mode'] == VISION_EXPERT_ADAPTATION:
        manifest['adaptation_mode'] = adaptation['mode']
    write_json(root / MANIFEST, manifest)
    result = verify_bundle(root, verify_base=False)
    result['base_sha256_verified'] = True
    return result


def verify_bundle(root, *, verify_base=True):
    root = Path(root).resolve()
    manifest_path = root / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    files = manifest.get('files')
    if (manifest.get('schema') not in (SCHEMA, RELATIVE_SCHEMA)
            or not isinstance(files, dict) or not REQUIRED.issubset(files)):
        raise ValueError('Invalid or incomplete stream checkpoint manifest')
    if 'prompt_integrity' in manifest:
        if (manifest['prompt_integrity'] != prompt_integrity.PROMPT_INTEGRITY
                or 'stream_prompt_integrity.py' not in files):
            raise ValueError('Checkpoint prompt integrity helper is missing or unsealed')
    elif (root / 'stream_prompt_integrity.py').exists():
        raise ValueError('Checkpoint has undeclared prompt integrity helper')
    if not any(name.startswith('tokenizer/') for name in files):
        raise ValueError('Missing tokenizer files')
    processor_files = {p.relative_to(root).as_posix() for p in root.glob('policy_*processor_step_*.safetensors')}
    if not processor_files or not processor_files.issubset(files):
        raise ValueError('Missing or unsealed processor statistics')
    for name, expected in files.items():
        path = safe_file(root, name)
        if (not isinstance(expected, dict) or path.stat().st_size != expected.get('bytes')
                or digest(path) != expected.get('sha256')):
            raise ValueError(f'Checkpoint file hash/size mismatch: {name}')
    report = json.loads(safe_file(root, 'training-report.json').read_text(encoding='utf-8'))
    if report.get('xyz_loss_mode') == 'terminal' and 'stream_terminal_loss.py' not in files:
        raise ValueError('Terminal XYZ checkpoint is missing its sealed sampling helper')
    if (report.get('dataset_fingerprint') != manifest.get('dataset_fingerprint')
            or report.get('steps_completed') != manifest.get('steps_completed')):
        raise ValueError('Checkpoint report disagrees with manifest')
    context_file = root / 'observation_context.json'
    if context_file.exists():
        if 'observation_context.json' not in files or 'stream_observation_history.py' not in files:
            raise ValueError('Unsealed observation context or helper')
        context = json.loads(context_file.read_text(encoding='utf-8'))
        if context != observation_history.context_metadata(context.get('profile')):
            raise ValueError('Invalid observation context metadata')
        context_profile = context['profile']
        if manifest.get('observation_context_profile') != context_profile:
            raise ValueError('Observation context disagrees with manifest')
    else:
        if 'observation_context_profile' in manifest:
            raise ValueError('Checkpoint lacks observation context metadata')
        context_profile = observation_history.NONE_PROFILE
    if report.get('observation_context_profile', observation_history.NONE_PROFILE) != context_profile:
        raise ValueError('Observation context disagrees with training report')
    config = json.loads(safe_file(root, 'config.json').read_text(encoding='utf-8'))
    validate_context_tokenizer(root, config, context_profile)
    n_action_steps = configured_action_steps(config, manifest['policy_contract']['chunk_size'])
    if ('n_action_steps' in report and (type(report['n_action_steps']) is not int
                                         or report['n_action_steps'] != n_action_steps)):
        raise ValueError('Checkpoint action execution steps disagree with training report')
    if ('n_action_steps' in manifest and (type(manifest['n_action_steps']) is not int
                                           or manifest['n_action_steps'] != n_action_steps)):
        raise ValueError('Checkpoint action execution steps disagree with manifest')
    image_file = root / 'image_profile.json'
    if image_file.exists():
        if 'image_profile.json' not in files or 'stream_image_profile.py' not in files:
            raise ValueError('Unsealed checkpoint image profile or helper')
    elif 'image_profile' in manifest:
        raise ValueError('Checkpoint lacks declared image-profile metadata')
    image_profile = read_image_profile(root)
    if manifest.get('image_profile', FULL_IMAGE_PROFILE) != image_profile['name']:
        raise ValueError('Checkpoint image profile disagrees with manifest')
    if (root / 'adaptation.json').exists() and 'adaptation.json' not in files:
        raise ValueError('Unsealed checkpoint adaptation metadata')
    adaptation = read_adaptation(root)
    if report.get('adaptation_mode', LORA_ADAPTATION) != adaptation['mode']:
        raise ValueError('Checkpoint adaptation disagrees with training report')
    if adaptation['mode'] in FULL_EXPERT_MODES and 'expert_model.safetensors' not in files:
        raise ValueError('Full-expert checkpoint is missing sealed expert weights')
    if adaptation['mode'] == VISION_EXPERT_ADAPTATION:
        if manifest.get('adaptation_mode') != VISION_EXPERT_ADAPTATION:
            raise ValueError('Checkpoint vision adaptation disagrees with manifest')
        if config.get('freeze_vision_encoder') is not False or config.get('train_expert_only') is not False:
            raise ValueError('Vision LoRA checkpoint must enable the vision path')
        adapter_config = json.loads(safe_file(root, 'adapter_config.json').read_text(encoding='utf-8'))
        if adapter_config.get('target_modules') != VISION_EXPERT_LORA_TARGETS:
            raise ValueError('Checkpoint adapter targets do not include the required vision q/v layers')
    if (root / 'action_representation.json').exists() and 'action_representation.json' not in files:
        raise ValueError('Unsealed checkpoint action-representation metadata')
    representation = read_action_representation(root)
    if (report.get('action_representation', ABSOLUTE_REPRESENTATION) != representation['mode']
            or manifest.get('action_representation', ABSOLUTE_REPRESENTATION) != representation['mode']):
        raise ValueError('Checkpoint action representation disagrees with manifest or training report')
    if ((representation['mode'] in RELATIVE_REPRESENTATIONS) != (manifest['schema'] == RELATIVE_SCHEMA)):
        raise ValueError('Checkpoint schema does not match action representation')
    if (representation['mode'] in RELATIVE_REPRESENTATIONS
            and 'relative_stream_actions.py' not in files):
        raise ValueError('Relative checkpoint is missing its sealed action converter')
    if representation['mode'] in CUP_REPRESENTATIONS and (
            context_profile not in observation_history.CUP_TCP_HISTORY_PROFILES or 'cup_relative_stream_data.py' not in files):
        raise ValueError('Cup-relative checkpoint lacks matching measured TCP context or data reader')
    if context_profile in observation_history.CUP_TRACKING_PROFILES and (
            not {'cup_tracking.py', 'rgbd_geometry.py'} <= files.keys()
            or not report.get('cup_tracking_data')):
        raise ValueError('Tracking checkpoint lacks the sealed observation implementation or coverage')
    if context_profile in observation_history.CUP_TIMING_PROFILES and 'cup_timing.py' not in files:
        raise ValueError('Timing checkpoint lacks its sealed physical state observer')
    if representation['mode'] in (CUP_DELTA_REPRESENTATION, CUP_TICK_DELTA_REPRESENTATION) and config.get('use_relative_actions') is not False:
        raise ValueError('Cup-relative delta checkpoint requires use_relative_actions=False')
    base = manifest.get('base', {})
    base_path = Path(base.get('weights_path', ''))
    if (not base_path.is_absolute() or not base_path.is_file()
            or base_path.stat().st_size != base.get('bytes') or not base.get('revision')):
        raise ValueError('Base weights missing or invalid')
    if verify_base and digest(base_path) != base.get('sha256'):
        raise ValueError('Base weights SHA256 mismatch')
    return {'checkpoint_sha256': digest(manifest_path), 'manifest': manifest,
            'adaptation_mode': adaptation['mode'], 'action_representation': representation['mode'],
            'wire_action_representation': (CUP_REPRESENTATION if representation['mode'] in CUP_REPRESENTATIONS
                                           else ABSOLUTE_REPRESENTATION),
            'n_action_steps': n_action_steps, 'image_profile': image_profile['name'],
            'observation_context_profile': context_profile,
            'base_sha256_verified': bool(verify_base)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.checkpoint), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
