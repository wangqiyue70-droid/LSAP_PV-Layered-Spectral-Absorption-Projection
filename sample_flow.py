"""Generate one image with a trained conditional Flow checkpoint and Euler ODE."""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from conditional_flow.data import EXTENSIONS, infer_style, read_condition
from conditional_flow.matching import euler_sample
from conditional_flow.model import ConditionalFlow, ModelConfig
from conditional_flow.runtime import device_from_name, load_checkpoint, sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="A train_flow.py checkpoint, not diffusion weights")
    parser.add_argument("--cond-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results_flow/sample.png"))
    parser.add_argument("--steps", type=int, default=20, help="Euler steps / network evaluations")
    parser.add_argument("--style-id", type=int, choices=range(4))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end with .png")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output or its JSON already exists; choose a new --output")
    try:
        condition_path = args.cond_path
        if condition_path.is_dir():
            paths = sorted(p for p in condition_path.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError(f"No condition images in {condition_path}")
            condition_path = paths[0]
        if not condition_path.is_file():
            raise ValueError(f"Missing condition: {condition_path}")
        style_id = args.style_id if args.style_id is not None else infer_style(condition_path)
        device = device_from_name(args.device)
        torch.set_num_threads(4)
        checkpoint = load_checkpoint(args.model_path)
        config = ModelConfig(**checkpoint["model_config"])
        model = ConditionalFlow(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        condition = read_condition(condition_path, config.image_size)[None].to(device)
        style = torch.tensor([style_id], dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(condition.shape, generator=generator, device=device)
        synthetic = checkpoint["data_info"]["kind"] == "synthetic_smoke"
        if synthetic:
            print("SYNTHETIC SMOKE CHECKPOINT: output is a software test, not a trained palm-vein model.", flush=True)
        print(f"Device={device} | Euler steps={args.steps} | style={style_id} | size={config.image_size} | "
              f"dino_mode={config.dino_mode} | teacher at inference=False", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        sample = euler_sample(model, noise, condition, style, args.steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        pixels = ((sample[0, 0] + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as f:
            Image.fromarray(pixels).save(f, format="PNG")
        metadata = dict(
            completed_at_utc=utc_now(), model_path=str(args.model_path.resolve()),
            checkpoint_sha256=sha256(args.model_path), checkpoint_step=checkpoint["step"],
            training_data_kind=checkpoint["data_info"]["kind"], synthetic_smoke_only=synthetic,
            model_config=config.to_dict(), device=str(device), seed=args.seed, style_id=style_id,
            sampler="Euler", nfe=args.steps, time_start=0, time_end=1, sampling_seconds=elapsed,
            condition=str(condition_path.resolve()), condition_sha256=sha256(condition_path),
            output=str(args.output.resolve()), output_sha256=sha256(args.output),
            torch=str(torch.__version__), cuda_build=torch.version.cuda,
            pixel_min=int(pixels.min()), pixel_max=int(pixels.max()), pixel_std=float(np.std(pixels)),
            dino_mode=config.dino_mode, feature_feedback=(config.dino_mode == "feedback"),
            teacher_required_at_inference=False, teacher_info=checkpoint.get("teacher_info"),
        )
        if device.type == "cuda":
            metadata.update(gpu=torch.cuda.get_device_name(device),
                            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
        write_json(args.output.with_suffix(".json"), metadata)
        print(f"Saved: {args.output.resolve()} | sampling seconds={elapsed:.3f}", flush=True)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
