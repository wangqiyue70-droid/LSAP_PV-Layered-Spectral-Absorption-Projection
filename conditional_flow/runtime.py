"""Flow-specific checkpoint format and local runtime helpers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import torch

FORMAT = "lsap_pv_conditional_flow_v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def device_from_name(name):
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f"CUDA device is unavailable: {device}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError("Not a conditional Flow checkpoint. Original diffusion weights cannot be sampled as Flow.")
    return checkpoint


def save_checkpoint(path, state):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        torch.save(state, f)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
