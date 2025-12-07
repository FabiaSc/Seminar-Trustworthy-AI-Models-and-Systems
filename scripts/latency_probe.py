# scripts/latency_probe.py
import argparse, csv, numpy as np, torch, yaml, os
from pathlib import Path
from omegaconf import OmegaConf

# Uses the same helpers as the existing txt2img/quant_txt2img scripts
from quant_utils.qdiff.utils import get_model, load_quant_params
from quant_utils.qdiff.models.quant_model import QuantModel

# Uses deterministic cuDNN settings for more stable latency measurements
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_pipe_fp16(config_yaml: str, fp16: bool = True):
    # Loads an fp16 UNet and diffusion pipeline and returns default sampling steps
    cfg = OmegaConf.load(config_yaml)
    model, pipe = get_model(cfg.model, fp16=fp16, return_pipe=True)
    return model, pipe, cfg.calib_data.n_steps


def build_pipe_w8a8(
    config_yaml: str,
    ckpt_path: str,
    weight_mp_cfg: str,
    act_mp_cfg: str,
    act_protect: str | None,
    fp16: bool = True,
):
    # Builds a quantized W8A8 (or WxAy) UNet and pipeline using mixed precision configs
    cfg = OmegaConf.load(config_yaml)

    # Creates model and pipeline ready for quantization
    model, pipe = get_model(
        cfg.model, fp16=fp16, return_pipe=True, convert_model_for_quant=True
    )

    # Prepares weight and activation quantizer configs, including optional mixed precision
    wq = cfg.quant.weight.quantizer
    aq = cfg.quant.activation.quantizer
    if cfg.get("mixed_precision", False):
        wq["mixed_precision"] = cfg.mixed_precision
        aq["mixed_precision"] = cfg.mixed_precision

    # Wraps the model in QuantModel on CUDA
    qnn = QuantModel(model=model, weight_quant_params=wq, act_quant_params=aq).cuda().eval()
    if fp16:
        qnn = qnn.half()

    # Enables weight and activation quantization and loads learned quant params
    qnn.set_quant_state(True, True)
    qnn.set_quant_init_done("weight")
    qnn.set_quant_init_done("activation")
    load_quant_params(qnn, ckpt_path, dtype=(torch.float16 if fp16 else torch.float32))

    # Applies activation protection if a list of modules is provided
    if act_protect:
        acts_protected = torch.load(act_protect)
        qnn.set_layer_quant(
            model=qnn,
            module_name_list=acts_protected,
            quant_level="per_layer",
            weight_quant=False,
            act_quant=False,
        )

    # Applies mixed precision bitwidth configs for weights and activations
    with open(weight_mp_cfg, "r") as f:
        qnn.load_bitwidth_config(qnn, yaml.safe_load(f), bit_type="weight")
    with open(act_mp_cfg, "r") as f:
        qnn.load_bitwidth_config(qnn, yaml.safe_load(f), bit_type="act")

    # Returns the quantized UNet, pipeline, and default number of steps
    return qnn, pipe, cfg.calib_data.n_steps


@torch.inference_mode()
def generate_once(pipe, unet, prompt: str, r: int, steps: int, cfg_scale: float):
    # Runs a single generation at resolution r with the given UNet and sampler settings
    assert r % 8 == 0, "Resolution must be multiple of 8"
    pipe.unet = unet.half() if next(unet.parameters()).dtype == torch.float16 else unet
    pipe.to("cuda")
    img = pipe(
        prompt=[prompt],
        height=r,
        width=r,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
    ).images[0]
    return img


def time_p50_p95(
    pipe,
    unet,
    prompt,
    r: int,
    steps: int,
    cfg_scale: float,
    repeats: int = 20,
    warmup: int = 2,
):
    # Measures p50 and p95 latency and peak memory for repeated generations
    for _ in range(warmup):
        _ = generate_once(pipe, unet, prompt, r, steps, cfg_scale)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        _ = generate_once(pipe, unet, prompt, r, steps, cfg_scale)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))  # ms

    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return p50, p95, mem_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp16", "w8a8", "w6a8", "w4a8"], required=True)
    ap.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to a run directory (contains config.yaml, ckpt.pth, images).",
    )
    ap.add_argument("--prompt", type=str, default="a corgi in sunglasses")
    ap.add_argument("--res", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("logs/latency_logs.csv"),
    )
    # Mixed precision config paths (same layout as shell scripts)
    ap.add_argument("--config_weight_mp", type=str, default=None)
    ap.add_argument("--config_act_mp", type=str, default=None)
    ap.add_argument("--act_protect", type=str, default=None)
    args = ap.parse_args()

    # Ensures the output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    config_yaml = os.path.join(args.base_path, "config.yaml")
    ckpt_path = os.path.join(args.base_path, "ckpt.pth")

    # Reads steps and cfg_scale from the YAML to match other scripts
    cfg = OmegaConf.load(config_yaml)
    steps = cfg.calib_data.n_steps
    cfg_scale = cfg.calib_data.scale_value

    # Builds either the fp16 or quantized pipeline, matching txt2img and quant_txt2img logic
    if args.mode == "fp16":
        unet, pipe, steps = build_pipe_fp16(config_yaml, fp16=True)
    else:
        unet, pipe, steps = build_pipe_w8a8(
            config_yaml,
            ckpt_path,
            args.config_weight_mp,
            args.config_act_mp,
            args.act_protect,
            fp16=True,
        )

    new = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["mode", "res", "p50_ms", "p95_ms", "peak_mem_GB", "prompt"],
        )
        if new:
            w.writeheader()
        for r in args.res:
            p50, p95, mem = time_p50_p95(
                pipe,
                unet,
                args.prompt,
                r,
                steps,
                cfg_scale,
                repeats=args.repeats,
            )
            w.writerow(
                {
                    "mode": args.mode.upper(),
                    "res": r,
                    "p50_ms": round(p50, 1),
                    "p95_ms": round(p95, 1),
                    "peak_mem_GB": round(mem, 3),
                    "prompt": args.prompt,
                }
            )


if __name__ == "__main__":
    main()
