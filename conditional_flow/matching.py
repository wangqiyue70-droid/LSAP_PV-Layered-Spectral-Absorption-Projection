"""Straight noise-to-data paths and explicit Euler integration."""
import torch
from torch.nn import functional as F


def flow_path(clean, noise, t):
    if clean.shape != noise.shape or t.shape != (clean.shape[0],):
        raise ValueError("Expected matching image/noise shapes and one time per sample")
    weight = t.reshape(-1, *([1] * (clean.ndim - 1)))
    return (1 - weight) * noise + weight * clean, clean - noise


def alignment_loss(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Predicted and teacher feature maps must have matching [B,D,H,W] shapes")
    return (1 - F.cosine_similarity(prediction.float(), target.detach().float(), dim=1, eps=1e-6)).mean()


def training_prediction(model, state, t, condition, style, self_cond_prob=0.9):
    """Use detached predictions, never clean teacher targets, as conditioning."""
    if not 0 <= self_cond_prob <= 1:
        raise ValueError("self_cond_prob must be in [0,1]")
    self_condition, used = None, 0.0
    if model.config.dino_mode == "feedback":
        mask = torch.rand(state.shape[0], device=state.device) < self_cond_prob
        if mask.any().item():
            # Autocast must not cache parameter casts made under no_grad and
            # then reuse these detached casts in the gradient-bearing pass.
            # This also keeps checkpoint recomputation metadata consistent.
            with torch.no_grad(), torch.autocast(
                    device_type=state.device.type,
                    enabled=torch.is_autocast_enabled(state.device.type),
                    dtype=torch.get_autocast_dtype(state.device.type), cache_enabled=False):
                preliminary = model.predict(state, t, condition, style).representation
            self_condition = preliminary.detach() * mask[:, None, None, None]
            used = mask.float().mean().item()
    return model.predict(state, t, condition, style, self_condition), used


@torch.no_grad()
def euler_sample(model, noise, condition, style, steps=20):
    if steps < 1:
        raise ValueError("steps must be positive")
    x = noise.clone()
    feedback = getattr(getattr(model, "config", None), "dino_mode", "none") == "feedback"
    self_condition = None
    for index in range(steps):
        t = torch.full((x.shape[0],), index / steps, device=x.device)
        if feedback:
            prediction = model.predict(x, t, condition, style, self_condition)
            velocity, self_condition = prediction.velocity, prediction.representation.detach()
            if not torch.isfinite(self_condition).all().item():
                raise FloatingPointError(f"Non-finite self-conditioning at Euler step {index}")
        else:
            velocity = model(x, t, condition, style)
        if velocity.shape != x.shape:
            raise ValueError("Velocity must have the same shape as the image")
        if not torch.isfinite(velocity).all().item():
            raise FloatingPointError(f"Non-finite velocity at Euler step {index}")
        x = x + velocity.float() / steps
    if not torch.isfinite(x).all().item():
        raise FloatingPointError("Non-finite final Flow sample")
    # Do not clamp intermediate states: this would change the ODE being solved.
    return x
