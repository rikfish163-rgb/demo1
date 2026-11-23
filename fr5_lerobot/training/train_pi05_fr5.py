"""Shared FR5 pi0.5 model, runtime and processor helpers."""

import hashlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/data4/hts/demo1/fr5-learning')


REVISION = '22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0'


TOKENIZER_REVISION = '35e4f46485b4d07967e7e9935bc3786aad50687c'


EXPECTED_WEIGHTS_SHA256 = '0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f'


IMAGE_KEY = 'observation.images.front'


MODEL_IMAGE_KEYS = ('observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb',
                    'observation.images.right_wrist_0_rgb')


TESTED_RUNTIME = {'lerobot': '0.6.1', 'torch': '2.11.0+cu128',
                  'transformers': '5.5.4', 'peft': '0.21.2'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_data4(path):
    path = Path(path).expanduser().resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError('Path must be inside data4/fr5-learning')
    return path


def require_runtime_versions():
    """Reject the local 0.4.4 exporter environment before importing ML code."""
    versions = {}
    for package in TESTED_RUNTIME:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = '<missing>'
    mismatches = [f'{name}={versions[name]} (need {wanted})'
                  for name, wanted in TESTED_RUNTIME.items() if versions[name] != wanted]
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        mismatches.append('interpreter is outside the tested data4 .venv')
    if mismatches:
        raise RuntimeError('Unsupported training runtime: ' + '; '.join(mismatches)
                           + f'. Run on server4090 with {ROOT}/.venv/bin/python')
    return versions


def configure_environment():
    """Pin caches and temporary files to data4 before importing ML libraries."""
    paths = {
        'HF_HOME': ROOT / 'cache/huggingface',
        'HF_HUB_CACHE': ROOT / 'cache/huggingface/hub',
        'HF_DATASETS_CACHE': ROOT / 'cache/huggingface/datasets',
        'HF_LEROBOT_HOME': ROOT / 'cache/huggingface/lerobot',
        'HF_XET_CACHE': ROOT / 'cache/xet',
        'TORCH_HOME': ROOT / 'cache/torch',
        'TORCHINDUCTOR_CACHE_DIR': ROOT / 'tmp/torchinductor',
        'TRITON_CACHE_DIR': ROOT / 'cache/triton',
        'CUDA_CACHE_PATH': ROOT / 'cache/cuda',
        'XDG_CACHE_HOME': ROOT / 'cache/xdg',
        'TMPDIR': ROOT / 'tmp',
    }
    for name, path in paths.items():
        check_data4(path).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                      WANDB_MODE='disabled')


def gpu_status(preferred=None):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, check=True, timeout=15)
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                '--format=csv,noheader,nounits'], capture_output=True,
                               text=True, check=True, timeout=15)
    owned, busy = set(), set()
    for line in processes.stdout.splitlines():
        if line.strip():
            uuid, pid = (value.strip() for value in line.split(','))
            (owned if int(pid) == os.getpid() else busy).add(uuid)
    candidates = []
    for line in result.stdout.splitlines():
        index, uuid, free, utilization = (value.strip() for value in line.split(','))
        index, free, utilization = int(index), int(free), int(utilization)
        if uuid not in busy and free >= 22000 and (utilization <= 5 or uuid in owned):
            candidates.append({'index': index, 'free_mib': free, 'utilization_pct': utilization})
    if preferred is not None:
        candidates = [candidate for candidate in candidates if candidate['index'] == preferred]
    if not candidates:
        raise RuntimeError('No requested GPU without compute processes and with at least 22000 MiB free')
    return max(candidates, key=lambda candidate: candidate['index'])


def register_padding_step():
    """Register the saved pipeline step before loading a trained processor."""
    import torch
    from lerobot.processor import ProcessorStepRegistry
    from lerobot.processor.pipeline import ProcessorStep, TransitionKey
    name = 'fr5_pi05_pad_normalized_state'
    if name in ProcessorStepRegistry.list():
        return ProcessorStepRegistry.get(name)

    @ProcessorStepRegistry.register(name=name)
    class PadNormalizedState(ProcessorStep):
        def __init__(self, max_state_dim=32, state_dim=8):
            self.max_state_dim = max_state_dim
            self.state_dim = state_dim

        def __call__(self, transition):
            transition = transition.copy()
            observation = dict(transition[TransitionKey.OBSERVATION])
            state = observation['observation.state']
            if state.ndim != 2 or state.shape[-1] != self.state_dim or self.state_dim > self.max_state_dim:
                raise ValueError('Expected the configured batched FR5 state dimension')
            observation['observation.state'] = torch.nn.functional.pad(state, (0, self.max_state_dim - self.state_dim))
            transition[TransitionKey.OBSERVATION] = observation
            return transition

        def get_config(self):
            return {'max_state_dim': self.max_state_dim, 'state_dim': self.state_dim}

        def transform_features(self, features):
            return features

    return PadNormalizedState


def make_processors(config, stats, tokenizer_path):
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import TokenizerProcessorStep, make_default_policy_processor_steps, make_policy_processor_pipelines
    default = make_default_policy_processor_steps(config, stats)
    default.rename_observations.rename_map = {IMAGE_KEY: MODEL_IMAGE_KEYS[0]}
    pad = register_padding_step()(max_state_dim=config.max_state_dim,
                                  state_dim=config.input_features['observation.state'].shape[0])
    return make_policy_processor_pipelines(
        input_steps=[default.rename_observations, default.add_batch_dim, default.normalize, pad,
                     Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
                     TokenizerProcessorStep(tokenizer_name=str(tokenizer_path), max_length=config.tokenizer_max_length,
                                            padding_side='right', padding='max_length'), default.to_device],
        output_steps=[default.unnormalize, default.to_cpu])


def tokenizer_path():
    return ROOT / 'cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots' / TOKENIZER_REVISION
