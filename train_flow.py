"""Train conditional Flow, optionally with DINO alignment and self-conditioning."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from conditional_flow.data import BatchStream, PairedVeinDataset, SyntheticSmokeDataset
from conditional_flow.matching import alignment_loss, flow_path, training_prediction
from conditional_flow.model import ConditionalFlow, ModelConfig
from conditional_flow.runtime import FORMAT, device_from_name, load_checkpoint, save_checkpoint, utc_now, write_json


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--data-dir", type=Path, help="Training root with paired vein/ and cond/ trees")
    data.add_argument("--smoke-test", action="store_true", help="Use synthetic toy pairs; NOT real training")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/flow"))
    parser.add_argument("--steps", type=int, default=1000, help="Total optimizer updates, including resumed updates")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--model-channels", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--style-id", type=int, choices=range(4))
    parser.add_argument("--smoke-size", type=int)
    parser.add_argument("--dino-mode", choices=("none", "align", "feedback"),
                        help="none: pure Flow; align: predict features; feedback: predict and self-condition")
    parser.add_argument("--dino-weight", type=float, help="Alignment loss coefficient (default: 0.5)")
    parser.add_argument("--self-cond-prob", type=float, help="Feedback probability per training sample (default: 0.9)")
    parser.add_argument("--dino-source", type=Path, help="Prepared local official DINOv2 source")
    parser.add_argument("--dino-weights", type=Path, help="Official ViT-S/14 teacher weights")
    parser.add_argument("--dino-device", help="Teacher device; defaults to the training device")
    return parser


def train(args):
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    saved_model = checkpoint["model_config"] if checkpoint else {}
    saved_train = checkpoint["train_config"] if checkpoint else {}

    def setting(name, default, saved):
        supplied = getattr(args, name)
        previous = saved.get(name, default)
        if checkpoint and supplied is not None and supplied != previous:
            raise ValueError(f"Cannot change {name} when resuming ({previous!r} -> {supplied!r})")
        return previous if supplied is None else supplied

    model_config = ModelConfig(**{
        name: setting(name, default, saved_model)
        for name, default in ModelConfig().to_dict().items()})
    train_config = {name: setting(name, default, saved_train) for name, default in dict(
        batch_size=1, grad_accum=1, lr=2e-4, seed=0, amp=False,
        augment=True, style_id=None, smoke_size=8,
        dino_weight=0.5, self_cond_prob=0.9).items()}
    if args.steps < 1 or args.save_every < 1:
        raise ValueError("steps and save-every must be positive")
    if min(train_config["batch_size"], train_config["grad_accum"], train_config["smoke_size"]) < 1:
        raise ValueError("batch-size, grad-accum and smoke-size must be positive")
    if not 0 < train_config["lr"] < float("inf"):
        raise ValueError("lr must be finite and positive")
    if not 0 <= train_config["self_cond_prob"] <= 1:
        raise ValueError("self-cond-prob must be in [0,1]")
    if model_config.dino_mode != "none" and not 0 < train_config["dino_weight"] < float("inf"):
        raise ValueError("DINO alignment weight must be finite and positive")
    if model_config.dino_mode == "none" and (args.dino_source or args.dino_weights or args.dino_device):
        raise ValueError("Teacher options require --dino-mode align or feedback")
    start_step = checkpoint["step"] if checkpoint else 0
    if args.steps <= start_step:
        raise ValueError(f"--steps is the TOTAL update count and must exceed the saved step {start_step}")
    device = device_from_name(args.device)
    if train_config["amp"] and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    torch.set_num_threads(4)
    torch.manual_seed(train_config["seed"])

    synthetic = args.smoke_test
    if checkpoint:
        was_synthetic = checkpoint["data_info"]["kind"] == "synthetic_smoke"
        if (args.smoke_test and not was_synthetic) or (args.data_dir and was_synthetic):
            raise ValueError("Cannot switch between synthetic and real data when resuming")
        synthetic = was_synthetic
    if synthetic:
        dataset = SyntheticSmokeDataset(model_config.image_size, train_config["smoke_size"], train_config["seed"])
        print("SYNTHETIC SMOKE TEST: software verification only; no real palm-vein training.", flush=True)
    else:
        root = args.data_dir or (checkpoint["data_info"]["root"] if checkpoint else None)
        if root is None:
            raise ValueError("Provide --data-dir for real paired data, or explicitly select --smoke-test")
        dataset = PairedVeinDataset(root, model_config.image_size, train_config["augment"], train_config["style_id"])
    if checkpoint and dataset.info["fingerprint"] != checkpoint["data_info"]["fingerprint"]:
        raise ValueError("Training data changed since the checkpoint; use a new run")

    teacher, teacher_info = None, None
    if model_config.dino_mode != "none":
        from conditional_flow.teacher import DEFAULT_SOURCE, DEFAULT_WEIGHTS, DinoTeacher
        previous_teacher = checkpoint.get("teacher_info") if checkpoint else None
        source = args.dino_source or (previous_teacher["source"] if previous_teacher else DEFAULT_SOURCE)
        weights = args.dino_weights or (previous_teacher["weights"] if previous_teacher else DEFAULT_WEIGHTS)
        teacher_device = args.dino_device or (previous_teacher.get("device", str(device)) if previous_teacher else str(device))
        teacher = DinoTeacher(source, weights, teacher_device)
        teacher_info = teacher.info
        if checkpoint and (not previous_teacher or previous_teacher["signature"] != teacher_info["signature"]):
            raise ValueError("DINO teacher source, weights or preprocessing changed; use a new run")

    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()) and not checkpoint:
        raise FileExistsError(f"Run directory is not empty: {output}. Use --resume or a new --out-dir.")
    final_path = output / f"flow_step_{args.steps:07d}.pt"
    if final_path.exists() or final_path.with_suffix(".json").exists():
        raise FileExistsError(f"Output already exists: {final_path}")
    output.mkdir(parents=True, exist_ok=True)
    model = ConditionalFlow(model_config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=train_config["amp"])
    stream = BatchStream(dataset, train_config["batch_size"], train_config["seed"])
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        stream.load_state_dict(checkpoint["stream_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
    del checkpoint

    print(f"Device={device} | size={model_config.image_size} | channels={model_config.model_channels} | "
          f"batch={train_config['batch_size']} | accumulation={train_config['grad_accum']} | "
          f"updates={start_step}->{args.steps} | dino_mode={model_config.dino_mode}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    last_loss, step, overflows = None, start_step, 0
    with (output / "train.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(dict(event="start", time=utc_now(), resumed_from=str(args.resume),
                                  model_config=model_config.to_dict(), train_config=train_config,
                                  data_info=dataset.info, teacher_info=teacher_info)) + "\n")
        log.flush()
        while step < args.steps:
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            flow_total, align_total, feedback_total = 0.0, 0.0, 0.0
            for _ in range(train_config["grad_accum"]):
                clean, condition, style = (value.to(device) for value in stream.next())
                noise = torch.randn_like(clean)
                t = torch.rand(clean.shape[0], device=device)
                state, target = flow_path(clean, noise, t)
                target_features = teacher(clean).to(device) if teacher is not None else None
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=train_config["amp"]):
                    if teacher is None:
                        prediction = model(state, t, condition, style)
                        feature_loss, feedback_used = torch.zeros((), device=device), 0.0
                    else:
                        result, feedback_used = training_prediction(
                            model, state, t, condition, style, train_config["self_cond_prob"])
                        prediction = result.velocity
                        feature_loss = alignment_loss(result.representation, target_features)
                    flow_loss = F.mse_loss(prediction.float(), target.float())
                    loss = flow_loss + train_config["dino_weight"]*feature_loss
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Flow loss is non-finite; no update saved")
                total_loss += loss.item() / train_config["grad_accum"]
                flow_total += flow_loss.item() / train_config["grad_accum"]
                align_total += feature_loss.item() / train_config["grad_accum"]
                feedback_total += feedback_used / train_config["grad_accum"]
                scaler.scale(loss / train_config["grad_accum"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not train_config["amp"] and not torch.isfinite(grad_norm).item():
                raise FloatingPointError("Non-finite gradients; no update saved")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if train_config["amp"] and scaler.get_scale() < previous_scale:
                overflows += 1
                log.write(json.dumps(dict(event="amp_overflow", step=step, scale=scaler.get_scale())) + "\n")
                log.flush()
                if overflows >= 20:
                    raise FloatingPointError("Repeated AMP overflow; retry a new run with --no-amp")
                continue
            overflows = 0
            step += 1
            last_loss = total_loss
            event = dict(event="update", step=step, loss=last_loss, flow_mse=flow_total,
                         alignment_loss=align_total, feedback_fraction=feedback_total, grad_norm=grad_norm.item())
            log.write(json.dumps(event) + "\n")
            log.flush()
            if step <= start_step + 3 or step % 10 == 0 or step == args.steps:
                print(f"Update {step}/{args.steps} | flow_mse={flow_total:.6f} | "
                      f"align={align_total:.6f} | feedback={feedback_total:.2f}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                state = dict(
                    format=FORMAT, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), model_config=model_config.to_dict(), train_config=train_config,
                    data_info=dataset.info, stream_state=stream.state_dict(),
                    teacher_info=teacher_info,
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=(torch.cuda.get_rng_state(device) if device.type == "cuda" else None),
                    last_loss=last_loss, torch_version=str(torch.__version__), created_at_utc=utc_now(),
                    time_convention="t=0: Gaussian noise; t=1: clean image; velocity=clean-noise",
                )
                save_checkpoint(output / f"flow_step_{step:07d}.pt", state)
                del state
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = dict(
        completed_at_utc=utc_now(), start_step=start_step, step=step,
        successful_updates_this_run=step-start_step, last_loss=last_loss,
        elapsed_seconds=time.perf_counter()-started, device=str(device),
        model_config=model_config.to_dict(), train_config=train_config, data_info=dataset.info,
        checkpoint=str(final_path), resumed_from=(str(args.resume.resolve()) if args.resume else None),
        torch=str(torch.__version__), cuda_build=torch.version.cuda,
        teacher_info=teacher_info,
        teacher_frozen=(all(not p.requires_grad and p.grad is None for p in teacher.parameters()) if teacher else None),
    )
    if device.type == "cuda":
        summary.update(gpu=torch.cuda.get_device_name(device),
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
    write_json(final_path.with_suffix(".json"), summary)
    print(f"Saved Flow checkpoint: {final_path}", flush=True)
    print(f"Training seconds: {summary['elapsed_seconds']:.2f}", flush=True)
    if device.type == "cuda":
        print(f"Peak PyTorch allocated GPU memory: {summary['peak_allocated_gib']:.3f} GiB", flush=True)
    return summary


def main():
    parser = create_parser()
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
