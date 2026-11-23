"""RTC guidance for the seven-dimensional FR5 joint stream."""
from __future__ import annotations

import torch
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.processor import TransitionKey


def normalize_absolute_prefix(old_abs, new_state, relative, normalizer):
    """Reanchor [N,7]/[B,N,7] targets to the matching [8]/[B,8] state.

    The tensors must share dtype and device; the returned prefix stays there.
    """
    if not isinstance(old_abs, torch.Tensor) or not isinstance(new_state, torch.Tensor):
        raise TypeError('Actions and state must be tensors')
    if (old_abs.ndim not in (2, 3) or new_state.ndim != old_abs.ndim - 1
            or old_abs.device != new_state.device or old_abs.dtype != new_state.dtype):
        raise ValueError('Actions and state must have matching batch, floating dtype and device')
    delta = relative.to_relative_actions(old_abs, new_state, input_representation=relative.ABSOLUTE)
    normalized = normalizer({TransitionKey.ACTION: delta})[TransitionKey.ACTION]
    if (normalized.shape != old_abs.shape or normalized.dtype != old_abs.dtype
            or normalized.device != old_abs.device):
        raise ValueError('Normalized prefix must preserve action shape, dtype and device')
    return normalized


class FR5RTCProcessor(RTCProcessor):
    """Apply RTC guidance to real action dimensions using the full VJP.

    A prefix has shape [N,7] or [B,N,7], matches the latent batch, dtype and
    device, and contains only its actual N steps. A two-dimensional prefix
    represents one batch. The caller must use no_grad rather than inference_mode.
    """

    def __init__(self, rtc_config, action_dim=7):
        if type(action_dim) is not int or action_dim != 7:
            raise ValueError('FR5 joint actions have exactly seven dimensions')
        super().__init__(rtc_config)
        self.action_dim = action_dim

    def denoise_step(self, x_t, prev_chunk_left_over, inference_delay, time,
                     original_denoise_step_partial, execution_horizon=None):
        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)
        if torch.is_inference_mode_enabled():
            raise RuntimeError('RTC guidance requires no_grad, not inference_mode')
        if not isinstance(x_t, torch.Tensor) or not isinstance(prev_chunk_left_over, torch.Tensor):
            raise TypeError('Latent and prefix must be tensors')
        if x_t.ndim not in (2, 3) or prev_chunk_left_over.ndim not in (2, 3):
            raise ValueError('Expected [T,A] or [B,T,A] tensors')
        squeezed = x_t.ndim == 2
        latent = x_t.unsqueeze(0) if squeezed else x_t
        prefix = prev_chunk_left_over.unsqueeze(0) if prev_chunk_left_over.ndim == 2 else prev_chunk_left_over
        if (latent.shape[0] != prefix.shape[0] or latent.shape[-1] < self.action_dim
                or prefix.shape[-1] != self.action_dim
                or not 0 < prefix.shape[1] <= latent.shape[1]):
            raise ValueError('Expected matching batches and a nonempty seven-dimensional prefix')
        if not latent.is_floating_point() or prefix.dtype != latent.dtype or prefix.device != latent.device:
            raise ValueError('Prefix and latent must have the same floating dtype and device')
        if type(inference_delay) is not int or inference_delay < 0:
            raise ValueError('inference_delay must be a nonnegative integer')
        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon
        if type(execution_horizon) is not int or execution_horizon < 0:
            raise ValueError('execution_horizon must be a nonnegative integer')
        horizon = min(execution_horizon, prefix.shape[1])
        weights = self.get_prefix_weights(inference_delay, horizon, latent.shape[1])
        weights = weights[:prefix.shape[1]].to(device=latent.device, dtype=latent.dtype)[None, :, None]
        step_time = torch.as_tensor(time, device=latent.device, dtype=latent.dtype)

        with torch.enable_grad():
            latent = latent.detach().clone().requires_grad_(True)
            velocity = original_denoise_step_partial(latent)
            clean = latent - step_time * velocity
            guided = clean[:, :prefix.shape[1], :self.action_dim]
            error = (prefix - guided) * weights
            correction = torch.autograd.grad(guided, latent, error.detach())[0]

        tau = 1 - step_time
        scale = (1 - tau) ** 2
        inv_r2 = (scale + tau ** 2) / scale
        maximum = self.rtc_config.max_guidance_weight
        c = torch.nan_to_num((1 - tau) / tau, posinf=maximum)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=maximum).clamp(max=maximum)
        result = velocity.detach() - guidance_weight * correction
        self.track(time=time, x1_t=clean.squeeze(0) if squeezed else clean,
                   correction=correction.squeeze(0) if squeezed else correction,
                   err=error.squeeze(0) if squeezed else error, weights=weights,
                   guidance_weight=guidance_weight, inference_delay=inference_delay,
                   execution_horizon=horizon)
        return result.squeeze(0) if squeezed else result
