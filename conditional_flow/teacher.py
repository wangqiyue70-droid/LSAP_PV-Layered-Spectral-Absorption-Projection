"""Frozen, offline DINOv2 ViT-S/14 teacher; never imported by sampling."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from .runtime import sha256

SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / ".downloads" / "dinov2"
DEFAULT_WEIGHTS = ROOT / ".downloads" / "dinov2_vits14_pretrain.pth"
FEATURE_DIM, FEATURE_GRID, INPUT_SIZE = 384, 16, 224


def source_fingerprint(source):
    digest = hashlib.sha256()
    files = [source / "hubconf.py", *sorted((source / "dinov2").rglob("*.py"))]
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class DinoTeacher(nn.Module):
    def __init__(self, source=DEFAULT_SOURCE, weights=DEFAULT_WEIGHTS, device="cpu"):
        super().__init__()
        source, weights = Path(source).resolve(), Path(weights).resolve()
        if not (source / "hubconf.py").is_file() or not weights.is_file():
            raise ValueError("DINOv2 source/weights are missing. Run setup_dino.ps1 in the research root first.")
        loaded = sys.modules.get("dinov2")
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
            raise ValueError("A different DINOv2 source is already imported; restart Python")
        sys.path.insert(0, str(source))
        try:
            factory = importlib.import_module("dinov2.hub.backbones").dinov2_vits14
            # Constructing the teacher must not change generator initialization
            # or the next batch's noise/time RNG, including on resume.
            with torch.random.fork_rng(devices=[]):
                self.encoder = factory(pretrained=False)
        finally:
            sys.path.pop(0)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state, strict=True)
        del state
        self.encoder.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        signature = dict(
            architecture="dinov2_vits14", weights_sha256=sha256(weights),
            source_sha256=source_fingerprint(source), feature_dim=FEATURE_DIM,
            feature_grid=FEATURE_GRID, input_size=INPUT_SIZE,
            preprocessing="gray[-1,1] to RGB[0,1]; bicubic antialias whole ROI 224; ImageNet normalization",
            feature_name="x_norm_patchtokens; last layer; no CLS or registers",
        )
        self.info = dict(signature=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
                         source=str(source), weights=str(weights), **signature)
        self.to(device)
        self.info["device"] = str(self.mean.device)
        self.train(False)

    def train(self, mode=True):
        # A caller cannot accidentally enable teacher dropout or stochastic depth.
        return super().train(False)

    @torch.no_grad()
    def forward(self, clean):
        if clean.ndim != 4 or clean.shape[1] != 1:
            raise ValueError("DINO expects a clean grayscale image batch [B,1,H,W]")
        self.encoder.eval()
        image = (clean.detach().to(self.mean.device).float() + 1) / 2
        image = image.clamp(0, 1).repeat(1, 3, 1, 1)
        image = F.interpolate(image, (INPUT_SIZE, INPUT_SIZE), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        with torch.autocast(device_type=self.mean.device.type, enabled=False):
            tokens = self.encoder.forward_features((image-self.mean)/self.std)["x_norm_patchtokens"]
        expected = (clean.shape[0], FEATURE_GRID*FEATURE_GRID, FEATURE_DIM)
        if tuple(tokens.shape) != expected or not torch.isfinite(tokens).all().item():
            raise FloatingPointError(f"Invalid DINO patch features: {tuple(tokens.shape)}")
        return tokens.transpose(1, 2).reshape(-1, FEATURE_DIM, FEATURE_GRID, FEATURE_GRID).contiguous()
