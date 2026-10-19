"""Reuse LSAP-PV's conditional U-Net, predicting velocity instead of noise."""
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from improved_diffusion.unet import SuperResModel
from improved_diffusion.nn import normalization, timestep_embedding


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 128
    model_channels: int = 64
    dropout: float = 0.2
    activation_checkpoint: bool = False
    dino_mode: str = "none"

    def __post_init__(self):
        if self.image_size not in (32, 64, 128):
            raise ValueError("image_size must be 32, 64 or 128")
        if self.model_channels < 32 or self.model_channels % 32:
            raise ValueError("model_channels must be a positive multiple of 32")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.dino_mode not in ("none", "align", "feedback"):
            raise ValueError("dino_mode must be none, align or feedback")

    def to_dict(self):
        return asdict(self)


class CheckpointBlock(nn.Module):
    """Modern checkpointing preserves autocast and dropout RNG on recomputation."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, embedding):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, x, embedding, use_reentrant=False)
        return self.block(x, embedding)


class FlowPrediction(NamedTuple):
    velocity: torch.Tensor
    representation: torch.Tensor


class RepresentationBackbone(SuperResModel):
    """Single spatial fusion in the encoder, feature prediction in the decoder."""
    feature_dim, feature_grid = 384, 16

    def __init__(self, *args, enable_feedback, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_feedback = enable_feedback
        channels = self.model_channels * 3
        self.representation_head = nn.Sequential(
            normalization(channels), nn.SiLU(), nn.Conv2d(channels, self.feature_dim, 1))
        if enable_feedback:
            self.feedback_projection = nn.Conv2d(self.feature_dim, channels, 1, bias=False)
            nn.init.normal_(self.feedback_projection.weight, std=0.01)

    def predict(self, image, time, condition, style, self_condition):
        x = torch.cat((image, condition), dim=1)
        embedding = self.time_embed(timestep_embedding(time, self.model_channels)) + self.label_emb(style)
        h, skips = x.type(self.inner_dtype), []
        fusion_resolution = image.shape[-1] // 8
        # Last residual block at the fourth encoder scale: 16x16 for 128x128.
        fusion_index = len(self.input_blocks) - self.num_res_blocks - 2
        for index, block in enumerate(self.input_blocks):
            h = block(h, embedding)
            if index == fusion_index and self_condition is not None:
                features = F.normalize(self_condition.float(), dim=1, eps=1e-6)
                features = F.interpolate(features, h.shape[-2:], mode="bilinear", align_corners=False)
                h = h + self.feedback_projection(features.to(h.dtype))
            skips.append(h)
        h = self.middle_block(h, embedding)
        decoder_feature = None
        for block in self.output_blocks:
            h = block(torch.cat((h, skips.pop()), dim=1), embedding)
            if h.shape[1] == self.model_channels * 3 and h.shape[-1] == fusion_resolution:
                decoder_feature = h
        if decoder_feature is None:
            raise RuntimeError("The DINO head could not find its decoder scale")
        representation = self.representation_head(decoder_feature.type(x.dtype))
        representation = F.interpolate(representation.float(), (self.feature_grid, self.feature_grid),
                                       mode="bilinear", align_corners=False)
        return FlowPrediction(self.out(h.type(x.dtype)), representation)


class ConditionalFlow(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        backbone = SuperResModel if config.dino_mode == "none" else RepresentationBackbone
        extra = {} if config.dino_mode == "none" else {"enable_feedback": config.dino_mode == "feedback"}
        self.backbone = backbone(
            in_channels=2, out_channels=1, model_channels=config.model_channels,
            num_res_blocks=2, attention_resolutions=(config.image_size // 4,),
            dropout=config.dropout, channel_mult=(1, 1, 2, 3, 4),
            num_classes=4, num_heads=4, num_heads_upsample=-1,
            use_checkpoint=False, use_scale_shift_norm=True,
            **extra,
        )
        if config.activation_checkpoint:
            self.backbone.input_blocks = nn.ModuleList(
                CheckpointBlock(block) for block in self.backbone.input_blocks)
            self.backbone.middle_block = CheckpointBlock(self.backbone.middle_block)
            self.backbone.output_blocks = nn.ModuleList(
                CheckpointBlock(block) for block in self.backbone.output_blocks)

    def _validate(self, x, t, condition, style):
        if x.shape != condition.shape or x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("x and condition must have identical [B, 1, H, W] shapes")
        if tuple(x.shape[-2:]) != (self.config.image_size, self.config.image_size):
            raise ValueError("Input size does not match the checkpoint model configuration")
        if t.shape != (x.shape[0],) or style.shape != t.shape:
            raise ValueError("t and style must have shape [B]")

    def predict(self, x, t, condition, style, self_condition=None):
        self._validate(x, t, condition, style)
        if self.config.dino_mode == "none":
            raise ValueError("Feature prediction requires dino_mode=align or feedback")
        if self_condition is not None:
            if self.config.dino_mode != "feedback":
                raise ValueError("Alignment-only mode does not accept self-conditioning")
            if self_condition.shape != (x.shape[0], 384, 16, 16):
                raise ValueError("Self-conditioning must have shape [B,384,16,16]")
            # Never propagate gradients between passes or solver steps.
            self_condition = self_condition.detach()
        return self.backbone.predict(x, t.float()*1000.0, condition, style, self_condition)

    def forward(self, x, t, condition, style):
        self._validate(x, t, condition, style)
        # t=0 is Gaussian noise; t=1 is data. Scaling only affects the sinusoidal
        # embedding and does not introduce diffusion steps or a noise schedule.
        if self.config.dino_mode != "none":
            return self.predict(x, t, condition, style).velocity
        return self.backbone(x, t.float() * 1000.0, cond=condition, y=style)
