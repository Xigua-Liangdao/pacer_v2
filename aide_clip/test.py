import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    from thop import profile
except ImportError:
    profile = None


class ClipImageAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
        num_prompts: int,
        use_prompt_weight: bool = True,
        use_class_temperature: bool = True,
        use_class_bias: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, dim)

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.prompt_weight_logits = nn.Parameter(torch.zeros(num_classes, num_prompts))
        self.class_logit_scale = nn.Parameter(torch.zeros(num_classes))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.use_prompt_weight = use_prompt_weight
        self.use_class_temperature = use_class_temperature
        self.use_class_bias = use_class_bias

    def _adapt_image(self, image_x: torch.Tensor) -> torch.Tensor:
        base = self.input_proj(image_x)
        delta = self.net(base)
        fused = base + delta
        img = self.out_proj(fused)
        return img / img.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def logits(self, image_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        img = self._adapt_image(image_x)
        txt = text_x / txt_norm(text_x)
        sim = torch.einsum("bd,cpd->bcp", img, txt)

        if self.use_prompt_weight:
            prompt_w = F.softmax(self.prompt_weight_logits, dim=-1).unsqueeze(0)
            class_sim = (sim * prompt_w).sum(dim=-1)
        else:
            class_sim = sim.mean(dim=-1)

        global_scale = self.logit_scale.exp().clamp(max=100.0)
        if self.use_class_temperature:
            class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).unsqueeze(0)
        else:
            class_scale = 1.0
        if self.use_class_bias:
            class_bias = self.class_bias.unsqueeze(0)
        else:
            class_bias = 0.0
        return global_scale * class_sim * class_scale + class_bias


def txt_norm(text_x: torch.Tensor) -> torch.Tensor:
    return text_x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline local-cache CLIP benchmark for the AIDE adapter setup")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--num-prompts", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-search-upper-bound", type=int, default=512)
    parser.add_argument("--plateau-tolerance", type=float, default=0.03)
    parser.add_argument("--plateau-rounds", type=int, default=3)
    parser.add_argument("--auto-start-frames", type=int, default=64)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_clip_offline(model_id: str, device: str):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(model_id, use_safetensors=False, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Offline CLIP load failed. Confirm the model exists in the local Hugging Face cache, for example ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32."
        ) from exc

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    return processor, model


def maybe_cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def build_dummy_inputs(processor, image_size: int, num_frames: int, device: str):
    blank = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    images = [blank.copy() for _ in range(num_frames)]
    return processor(images=images, return_tensors="pt", padding=True).to(device)


def manual_adapter_flops(dim: int, hidden_dim: int) -> int:
    return (dim * hidden_dim + hidden_dim * hidden_dim * 2 + hidden_dim * dim) * 2


def measure_raw_clip_case(model, device: str, image_size: int, num_frames: int, warmup_iters: int, num_iters: int):
    torch.cuda.empty_cache()
    try:
        dummy = torch.randn(num_frames, 3, image_size, image_size, device=device, dtype=torch.float16)
        for _ in range(warmup_iters):
            with torch.no_grad():
                _ = model.get_image_features(pixel_values=dummy)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                _ = model.get_image_features(pixel_values=dummy)
            torch.cuda.synchronize()
        elapsed_ms = (time.time() - start_time) / num_iters * 1000.0
        clips_per_sec = 1000.0 / elapsed_ms if elapsed_ms > 0 else float("inf")
        frames_per_sec = num_frames * clips_per_sec
        del dummy
        return {
            "frames": num_frames,
            "status": "OK",
            "ms_per_clip": elapsed_ms,
            "clips_per_sec": clips_per_sec,
            "frames_per_sec": frames_per_sec,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "frames": num_frames,
            "status": "OOM",
            "ms_per_clip": None,
            "clips_per_sec": None,
            "frames_per_sec": None,
        }


def print_raw_clip_table(rows, stop_reason: str) -> None:
    print("\nRaw CLIP scaling table")
    print(f"{'frames':>8} {'status':>8} {'ms/clip':>12} {'clips/s':>12} {'frames/s':>12} {'delta':>10}")
    print("-" * 70)
    prev_fps = None
    for row in rows:
        if row["status"] != "OK":
            print(f"{row['frames']:>8} {row['status']:>8} {'-':>12} {'-':>12} {'-':>12} {'-':>10}")
            continue
        delta_text = "-"
        if prev_fps is not None and prev_fps > 0:
            delta = (row["frames_per_sec"] - prev_fps) / prev_fps * 100.0
            delta_text = f"{delta:+.2f}%"
        print(
            f"{row['frames']:>8} {row['status']:>8} {row['ms_per_clip']:>12.1f} "
            f"{row['clips_per_sec']:>12.2f} {row['frames_per_sec']:>12.2f} {delta_text:>10}"
        )
        prev_fps = row["frames_per_sec"]
    print(f"Stop reason: {stop_reason}")


def benchmark_raw_clip_scaling(
    model,
    device: str,
    image_size: int,
    warmup_iters: int,
    num_iters: int,
    start_frames: int,
    initial_upper_bound: int,
    plateau_tolerance: float,
    plateau_rounds: int,
):
    if not device.startswith("cuda"):
        print("\nRaw CLIP scaling table: skipped because device is not CUDA")
        return

    seed_points = [1, 3, 5, 10, 16, 32, 64]
    rows = []
    plateau_hits = 0
    last_ok_frames = 0
    last_tested = 0
    previous_ok_fps = None
    stop_reason = "reached configured upper bound without OOM or plateau"

    print("\nAuto-expanding raw CLIP benchmark...")

    for num_frames in seed_points:
        row = measure_raw_clip_case(model, device, image_size, num_frames, warmup_iters, num_iters)
        rows.append(row)
        last_tested = num_frames
        if row["status"] == "OOM":
            stop_reason = f"OOM at {num_frames} frames"
            print_raw_clip_table(rows, stop_reason)
            return
        last_ok_frames = num_frames
        previous_ok_fps = row["frames_per_sec"]

    next_frames = max(start_frames, seed_points[-1] * 2)
    current_upper_bound = max(initial_upper_bound, next_frames)

    while True:
        while next_frames > current_upper_bound:
            current_upper_bound *= 2

        row = measure_raw_clip_case(model, device, image_size, next_frames, warmup_iters, num_iters)
        rows.append(row)
        last_tested = next_frames

        if row["status"] == "OOM":
            stop_reason = f"OOM at {next_frames} frames"
            break

        last_ok_frames = next_frames
        if previous_ok_fps is not None and previous_ok_fps > 0:
            relative_change = abs(row["frames_per_sec"] - previous_ok_fps) / previous_ok_fps
            if relative_change <= plateau_tolerance:
                plateau_hits += 1
            else:
                plateau_hits = 0
        previous_ok_fps = row["frames_per_sec"]

        if plateau_hits >= plateau_rounds:
            stop_reason = (
                f"frame throughput plateaued within {plateau_tolerance * 100:.1f}% for {plateau_rounds} consecutive expansion steps"
            )
            break

        next_frames *= 2

    print_raw_clip_table(rows, stop_reason)
    print(f"Largest successful raw CLIP forward: {last_ok_frames} frames")
    print(f"Last tested frame count: {last_tested}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    processor, clip_model = load_clip_offline(args.model_id, device)

    adapter = ClipImageAdapter(
        dim=int(clip_model.config.projection_dim),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_classes=args.num_classes,
        num_prompts=args.num_prompts,
    ).to(device).eval()

    backbone_params = sum(p.numel() for p in clip_model.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters())
    total_params = backbone_params + trainable_params

    print(f"Model ID: {args.model_id}")
    print(f"Device: {device}")
    print("Load mode: local cache only")
    print(f"Backbone (frozen): {backbone_params / 1e6:.2f}M")
    print(f"Trainable (adapter): {trainable_params / 1e6:.2f}M")
    print(f"Total: {total_params / 1e6:.2f}M")

    dummy_img = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    flops_clip = None
    if profile is not None:
        flops_clip, _ = profile(clip_model.vision_model, inputs=(dummy_img,), verbose=False)
        print(f"\nFLOPs (backbone per frame): {flops_clip / 1e9:.2f}G")
    else:
        print("\nFLOPs (backbone per frame): unavailable, thop is not installed in this environment")

    flops_adapter = manual_adapter_flops(int(clip_model.config.projection_dim), args.hidden_dim)
    print(f"FLOPs (adapter): {flops_adapter / 1e6:.2f}M")
    if flops_clip is not None:
        flops_total = flops_clip * args.num_frames + flops_adapter
        print(f"FLOPs (total, {args.num_frames} frames): {flops_total / 1e9:.2f}G")

    inputs = build_dummy_inputs(processor, args.image_size, args.num_frames, device)
    text_features = torch.randn(
        args.num_classes,
        args.num_prompts,
        int(clip_model.config.projection_dim),
        device=device,
        dtype=clip_model.visual_projection.weight.dtype,
    )
    text_features = text_features / txt_norm(text_features)

    with torch.no_grad():
        for _ in range(args.warmup):
            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            pooled = image_features.mean(dim=0, keepdim=True)
            _ = adapter.logits(pooled.float(), text_features.float())

    maybe_cuda_sync(device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.iters):
            image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            pooled = image_features.mean(dim=0, keepdim=True)
            _ = adapter.logits(pooled.float(), text_features.float())
            maybe_cuda_sync(device)
    t1 = time.time()

    ms_per_sample = (t1 - t0) / args.iters * 1000.0
    fps = 1000.0 / ms_per_sample if ms_per_sample > 0 else float("inf")
    print(f"\nInference: {ms_per_sample:.1f} ms/sample, {fps:.2f} FPS")
    print(f"(includes {args.num_frames} frames through backbone + adapter)")

    benchmark_raw_clip_scaling(
        clip_model,
        device,
        args.image_size,
        warmup_iters=max(1, min(5, args.warmup)),
        num_iters=max(5, min(50, args.iters)),
        start_frames=args.auto_start_frames,
        initial_upper_bound=args.max_search_upper_bound,
        plateau_tolerance=args.plateau_tolerance,
        plateau_rounds=args.plateau_rounds,
    )


if __name__ == "__main__":
    main()
