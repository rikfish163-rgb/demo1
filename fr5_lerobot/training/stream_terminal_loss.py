"""Differentiable pi05 terminal sampling for cup translation supervision."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@torch.no_grad()
def move_optimizer_moments(optimizer, *, offload: bool):
    """Move AdamW moments to CPU or their parameter's device without changing step."""
    moment_bytes = 0
    devices = set()
    for parameter, state in optimizer.state.items():
        target = torch.device('cpu') if offload else parameter.device
        for name in ('exp_avg', 'exp_avg_sq'):
            moment = state.get(name)
            if moment is None:
                continue
            if moment.device != target:
                moment = moment.to(device=target)
                state[name] = moment
            moment_bytes += moment.numel() * moment.element_size()
            devices.add(str(moment.device))
    return {'moment_bytes': moment_bytes, 'devices': sorted(devices)}


@contextmanager
def sampling_mode(policy):
    """Use deployment module modes while retaining all trainable parameters."""
    states = [(module, module.training) for module in policy.modules()]
    policy.eval()
    try:
        yield
    finally:
        for module, state in states:
            module.training = state


def sample_terminal_actions(policy, batch, noise):
    """Use the native complete Euler sampler with a retained gradient graph."""
    from train_stream_pi05 import ACTION_DIM, action_clock_condition

    model = policy.model
    native_sample = model.sample_actions.__wrapped__
    native_denoise = model.denoise_step
    steps = policy.config.num_inference_steps

    def denoise(*, prefix_pad_masks, past_key_values, x_t, timestep):
        def step(state, time):
            with sampling_mode(policy), action_clock_condition(policy, batch):
                return native_denoise(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                                      x_t=state, timestep=time)
        return checkpoint(step, x_t, timestep, use_reentrant=False)

    def integrate(initial_noise):
        with sampling_mode(policy), action_clock_condition(policy, batch):
            original = model.denoise_step
            model.denoise_step = denoise
            try:
                images, image_masks = policy._preprocess_images(batch)
                sampled = native_sample(model, images, image_masks,
                    batch['observation.language.tokens'], batch['observation.language.attention_mask'],
                    noise=initial_noise, num_steps=steps)
                return sampled[..., :ACTION_DIM]
            finally:
                model.denoise_step = original

    return checkpoint(integrate, noise.detach().clone(), use_reentrant=False)


def terminal_xyz_penalty(error, action_scale, weights):
    """Measure the sampled prefix translation error with the original mm scale."""
    ticks = torch.arange(1, error.shape[1] + 1, device=error.device)
    error_mm = error[..., :3] * action_scale[:3] * ticks[:, None] * 1000.
    radius_mm = error_mm.square().sum(-1).add(1e-8).sqrt()
    excess = (radius_mm - .25).clamp_min(0.)
    return (6. * F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction='none') * weights[..., 0]).sum()
