"""Sample one image with the original LSAP-PV checkpoint, without MPI."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--cond_path", type=Path, required=True,
                        help="Condition image, or directory (uses its first image).")
    parser.add_argument("--output", type=Path, default=Path("results_smoke/sample.png"))
    parser.add_argument("--batch_size", type=int, choices=[1], default=1)
    parser.add_argument("--style_id", type=int, choices=range(4), default=None,
                        help="0=casia, 1=HFUT, 2=polyu, 3=TongJi; otherwise read folder suffix.")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (default) or cpu (slow).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.model_path.is_file():
        parser.error(f"Checkpoint not found: {args.model_path}. Download and unzip DM-ckp first.")
    cond_path = args.cond_path
    if cond_path.is_dir():
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        images = sorted(p for p in cond_path.rglob("*")
                        if p.is_file() and p.suffix.lower() in extensions)
        if not images:
            parser.error(f"No condition images found in {cond_path}")
        cond_path = images[0]
    if not cond_path.is_file():
        parser.error(f"Condition image not found: {cond_path}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a different --output.")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")

    style = args.style_id
    if style is None:
        suffix = cond_path.parent.name.rsplit("_", 1)[-1]
        if suffix not in {"0", "1", "2", "3"}:
            parser.error("Cannot infer style from condition folder; specify --style_id 0, 1, 2 or 3.")
        style = int(suffix)

    try:
        import numpy as np
        from PIL import Image
        import torch
        import tqdm  # Fail before loading the model if the progress dependency is absent.
        from improved_diffusion.script_util import (
            vein_create_model_and_diffusion,
            vein_model_and_diffusion_defaults,
        )
    except ImportError as exc:
        parser.error(f"Missing dependency: {exc}. See RUN_SINGLE_GPU.md.")

    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("Use --device cuda:0 or --device cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Check your PyTorch/CUDA installation, or use --device cpu.")
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        parser.error(f"CUDA device does not exist: {device}")
    if device.type == "cpu":
        print("CPU selected: original 1000-step sampling can be very slow.", flush=True)

    # Match CondDataset: grayscale, PIL bilinear resize, ToTensor, Normalize(0.5, 0.5).
    with Image.open(cond_path) as image:
        image = image.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32) / 255.0
    condition = torch.from_numpy((array - 0.5) / 0.5)[None, None].to(device)
    labels = torch.tensor([style], dtype=torch.long, device=device)

    torch.manual_seed(args.seed)
    # Exactly the architecture and schedule used by the original sample.sh.
    config = vein_model_and_diffusion_defaults()
    config.update(learn_sigma=True, class_cond=True)
    print(f"Checkpoint: {args.model_path.resolve()}", flush=True)
    print(f"Condition: {cond_path.resolve()} | style_id={style}", flush=True)
    print(f"Device: {device} | batch_size=1 | image_size=128 | DDPM steps=1000", flush=True)
    model, diffusion = vein_create_model_and_diffusion(**config)
    state = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()

    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model, (1, 1, 128, 128), clip_denoised=True,
            model_kwargs={"cond": condition, "y": labels},
            device=device, progress=True,
        )
    if not torch.isfinite(sample).all().item():
        raise RuntimeError("Sampling produced non-finite values; no image was saved.")
    pixels = ((sample[0, 0] + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        Image.fromarray(pixels).save(output, format="PNG")
    print(f"Saved: {args.output.resolve()}", flush=True)
    if device.type == "cuda":
        print(f"Peak PyTorch allocated GPU memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
