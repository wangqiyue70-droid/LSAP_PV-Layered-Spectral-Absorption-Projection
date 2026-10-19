"""Strict paired data loading without MPI or blobfile."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STYLES = {"casia": 0, "hfut": 1, "polyu": 2, "tongji": 3}


def infer_style(path):
    name = Path(path).parent.name
    prefix = name.split("_", 1)[0].casefold()
    if prefix in STYLES:
        return STYLES[prefix]
    suffix = name.rsplit("_", 1)[-1]
    if suffix in {"0", "1", "2", "3"}:
        return int(suffix)
    raise ValueError(f"Cannot infer style from {name!r}; supply --style-id 0, 1, 2 or 3")


def image_tensor(image):
    array = np.array(image, dtype=np.float32, copy=True) / 127.5 - 1.0
    return torch.from_numpy(array)[None]


def read_condition(path, image_size):
    with Image.open(path) as image:
        return image_tensor(image.convert("L").resize(
            (image_size, image_size), Image.Resampling.BILINEAR))


def image_index(directory):
    if not directory.is_dir():
        raise ValueError(f"Missing data directory: {directory}")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            key = path.relative_to(directory).with_suffix("").as_posix().casefold()
            if key in result:
                raise ValueError(f"Ambiguous duplicate sample key: {key}")
            result[key] = path
    if not result:
        raise ValueError(f"No images found in {directory}")
    return result


class PairedVeinDataset(Dataset):
    def __init__(self, root, image_size=128, augment=True, style_id=None):
        self.root = Path(root).resolve()
        self.image_size, self.augment = image_size, augment
        images = image_index(self.root / "vein")
        conditions = image_index(self.root / "cond")
        if images.keys() != conditions.keys():
            missing_cond = sorted(images.keys() - conditions.keys())[:5]
            missing_image = sorted(conditions.keys() - images.keys())[:5]
            raise ValueError(f"Unpaired data: missing conditions={missing_cond}; missing images={missing_image}")
        self.pairs = [(images[key], conditions[key],
                       infer_style(images[key]) if style_id is None else style_id)
                      for key in sorted(images)]
        inventory = [(str(p.relative_to(self.root)), p.stat().st_size, p.stat().st_mtime_ns)
                     for image, condition, _ in self.pairs for p in (image, condition)]
        fingerprint = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
        self.info = dict(kind="real_paired", root=str(self.root), count=len(self),
                         fingerprint=fingerprint, fingerprint_scope="relative paths, sizes and modification times",
                         style_override=style_id)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, cond_path, style = self.pairs[index]
        with Image.open(image_path) as source, Image.open(cond_path) as cond_source:
            if source.size != cond_source.size:
                raise ValueError(f"Pair has different spatial sizes: {image_path}, {cond_path}")
            image, condition = source.convert("L"), cond_source.convert("L")
        size = int(self.image_size * 1.1) if self.augment else self.image_size
        if self.augment:
            brightness, contrast = (0.99 + 0.02 * torch.rand(2)).tolist()
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        condition = condition.resize((size, size), Image.Resampling.BILINEAR)
        if self.augment:
            left, top = torch.randint(size - self.image_size + 1, (2,)).tolist()
            box = (left, top, left + self.image_size, top + self.image_size)
            image, condition = image.crop(box), condition.crop(box)
        return image_tensor(image), image_tensor(condition), style


class SyntheticSmokeDataset(Dataset):
    """Procedural toy pairs; never a real palm-vein training dataset."""
    def __init__(self, image_size=128, count=8, seed=0):
        self.image_size, self.count, self.seed = image_size, count, seed
        if count < 1:
            raise ValueError("Synthetic sample count must be positive")
        self.info = dict(kind="synthetic_smoke", count=count, seed=seed,
                         fingerprint=f"toy-veins-v1:{image_size}:{count}:{seed}",
                         purpose="software verification only; no real biometric data")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + index)
        axis = torch.linspace(-1, 1, self.image_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        vessel = torch.zeros_like(xx)
        for _ in range(3):
            slope, offset = (torch.rand(2, generator=generator) - 0.5).tolist()
            distance = xx - slope * yy - offset - 0.12 * torch.sin(4 * yy + offset)
            vessel = torch.maximum(vessel, torch.exp(-distance.square() / 0.0015))
        style = index % 4
        condition = (1 - 2 * vessel)[None]
        image = (0.15 + 0.04 * style - 0.45 * vessel + 0.06 * yy
                 + 0.01 * torch.randn(xx.shape, generator=generator)).clamp(-1, 1)[None]
        return image, condition, style


class BatchStream:
    """A checkpointable shuffle cursor; no worker prefetch or hidden RNG state."""
    def __init__(self, dataset, batch_size, seed):
        self.dataset, self.batch_size = dataset, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.cursor = [], 0

    def next(self):
        batch = []
        for _ in range(self.batch_size):
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
                self.cursor = 0
            batch.append(self.dataset[self.order[self.cursor]])
            self.cursor += 1
        clean, condition, style = zip(*batch)
        return torch.stack(clean), torch.stack(condition), torch.tensor(style, dtype=torch.long)

    def state_dict(self):
        return dict(order=self.order, cursor=self.cursor, generator=self.generator.get_state())

    def load_state_dict(self, state):
        order, cursor = state["order"], state["cursor"]
        if sorted(order) != list(range(len(self.dataset))) or not 0 <= cursor <= len(order):
            raise ValueError("Invalid or incompatible saved data cursor")
        self.order, self.cursor = order, cursor
        self.generator.set_state(state["generator"])
