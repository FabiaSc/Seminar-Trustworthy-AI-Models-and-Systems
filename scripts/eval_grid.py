#!/usr/bin/env python3
# scripts/eval_grid.py

import os, csv, time, argparse, yaml, random, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

# Tries to import qdiff from a local quant_utils namespace, then falls back to qdiff
try:
    from quant_utils.qdiff.utils import get_model, load_quant_params
    from quant_utils.qdiff.models.quant_model import QuantModel
except Exception:
    from qdiff.utils import get_model, load_quant_params
    from qdiff.models.quant_model import QuantModel

# Uses deterministic cuDNN settings for more stable benchmarks
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------ CLIP (CPU/FP32) with logging ------------------------
_CLIP_READY = False
_CLIP_MODEL = None
_CLIP_PREPROC = None


def _ensure_clip_ready():
    # Lazily loads an open_clip ViT-B-32 model on CPU in float32
    global _CLIP_READY, _CLIP_MODEL, _CLIP_PREPROC
    if _CLIP_READY:
        return True
    try:
        import open_clip
        print("[CLIP] loading open_clip ViT-B-32 (laion2b_s34b_b79k) on CPU/FP32...")
        _CLIP_MODEL, _, _CLIP_PREPROC = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _CLIP_MODEL.eval().to("cpu")
        _CLIP_READY = True
        print("[CLIP] model ready.")
        return True
    except Exception as e:
        print("[CLIP][ERROR] cannot load model:", e)
        traceback.print_exc()
        return False


def clipscore(prompts, images):
    """
    Returns the mean CLIP cosine similarity between images and texts.
    Runs open_clip on CUDA in fp16 when available, otherwise on CPU in fp32.
    """
    if not images:
        print("[CLIP][WARN] no images to score (images list empty).")
        return float("nan")
    if not _ensure_clip_ready():
        print("[CLIP][WARN] CLIP model not available, returning NaN.")
        return float("nan")

    try:
        import open_clip
    except Exception as e:
        print("[CLIP][ERROR] open_clip not importable:", e)
        return float("nan")

    # Selects device and optionally switches the model to fp16 on CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _CLIP_MODEL.to(device)
    _CLIP_MODEL.eval()
    use_fp16 = device.type == "cuda"
    if use_fp16:
        _CLIP_MODEL.half()
        print("[CLIP] running CLIP in FP16 on CUDA")
    else:
        print("[CLIP] CUDA not available; running CLIP in FP32 on CPU")

    sims = []
    n_imgs = min(len(prompts), len(images))

    with torch.no_grad():
        for i in range(n_imgs):
            p = prompts[i]
            img = images[i]
            try:
                # Normalizes image input to a PIL instance
                if isinstance(img, Image.Image):
                    pil = img
                else:
                    pil = Image.fromarray(np.array(img))

                im = _CLIP_PREPROC(pil).unsqueeze(0)
                im = im.to(device)
                if use_fp16:
                    im = im.half()

                # Prepares and tokenizes the text for CLIP
                tok = open_clip.tokenize([p]).to(device)

                # Encodes image and text to CLIP embeddings
                img_emb = _CLIP_MODEL.encode_image(im)
                txt_emb = _CLIP_MODEL.encode_text(tok)

                # Casts back to float32 for normalization and dot product
                img_emb = img_emb.float()
                txt_emb = txt_emb.float()

                img_n = F.normalize(img_emb, dim=-1, eps=1e-6)
                txt_n = F.normalize(txt_emb, dim=-1, eps=1e-6)
                sim = (img_n * txt_n).sum(dim=-1).squeeze()

                if torch.isfinite(sim):
                    sims.append(sim.item())
                else:
                    print(f"[CLIP][WARN] non-finite sim for sample {i}.")
            except Exception as e:
                print(f"[CLIP][WARN] scoring failed for sample {i}:", e)
                traceback.print_exc()

    n_valid = len(sims)
    print(f"[CLIP] n_imgs={n_imgs} n_valid={n_valid}")
    return float("nan") if n_valid == 0 else float(sum(sims) / n_valid)


# ------------------------ Quantized model (W N bits, A16) ------------------------
def build_qnn(config_yaml: str, ckpt_path: str, fp16: bool = True):
    """
    Loads SDXL-Turbo converted for quantization and returns:
    (config, diffusion pipeline, quantized UNet, default steps, guidance scale).
    """
    cfg = OmegaConf.load(config_yaml)

    # Requests a UNet that is ready to be wrapped by QuantModel
    unet, pipe = get_model(cfg.model, fp16=fp16, return_pipe=True, convert_model_for_quant=True)

    wq = cfg.quant.weight.quantizer
    aq = cfg.quant.activation.quantizer
    # Propagates optional mixed precision settings into quantizer configs
    if cfg.get("mixed_precision", False):
        wq["mixed_precision"] = cfg.mixed_precision
        aq["mixed_precision"] = cfg.mixed_precision

    # Wraps the UNet in QuantModel with weight and activation quantization parameters
    qnn = QuantModel(model=unet, weight_quant_params=wq, act_quant_params=aq).cuda().eval()
    if fp16:
        qnn = qnn.half()

    # Enables quantized weights and keeps activations in fp16 for stability
    qnn.set_quant_state(True, False)
    qnn.set_quant_init_done("weight")
    qnn.set_quant_init_done("activation")

    dtype = torch.float16 if fp16 else torch.float32
    load_quant_params(qnn, ckpt_path, dtype=dtype)

    steps = int(cfg.calib_data.n_steps)
    guide = float(cfg.calib_data.scale_value)
    return cfg, pipe, qnn, steps, guide


def apply_weight_bits(qnn: QuantModel, weight_yaml_path: str):
    # Loads a bitwidth configuration from YAML and applies it to the quantized UNet
    with open(weight_yaml_path, "r") as f:
        bit_cfg = yaml.safe_load(f)
    qnn.load_bitwidth_config(model=qnn, bit_config=bit_cfg, bit_type="weight")


# ------------------------ Main sweep ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="./configs/stable-diffusion/sdxl_turbo.yaml",
        help="Path to SDXL config YAML.",
    )
    ap.add_argument(
        "--ckpt",
        default="./logs/sdxl_mixdq_eval/ckpt.pth",
        help="Path to ckpt.pth with quant params.",
    )
    ap.add_argument(
        "--weight_dir",
        default="./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight",
        help="Directory with weight bitwidth YAML files.",
    )
    ap.add_argument(
        "--out_csv",
        default="logs/grid_bits_res.csv",
        help="Output CSV file for metrics.",
    )
    ap.add_argument(
        "--wbits",
        type=int,
        nargs="+",
        default=[8, 6, 4],
        help="List of weight bitwidths to evaluate.",
    )
    ap.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[1024, 768, 512],
        help="Resolutions to test (must be multiples of 8).",
    )
    ap.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=[
            "a corgi in sunglasses",
            "a red car on a snowy road",
            "a bowl of ramen on a wooden table",
            "a modern living room with plants",
            "a medieval castle at sunrise",
        ],
    )
    ap.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a text file with one prompt per line; overrides --prompts.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Overrides the prompts list when a prompts file is provided
    if args.prompts_file is not None:
        if not os.path.isfile(args.prompts_file):
            raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            raise ValueError(f"Prompts file {args.prompts_file} is empty.")
        args.prompts = lines
        print(f"[INFO] loaded {len(args.prompts)} prompts from {args.prompts_file}")

    os.makedirs("logs", exist_ok=True)
    for r in args.res:
        assert r % 8 == 0, f"Resolution must be multiple of 8, got {r}"

    # Seeds random generators for lightweight reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Loads config, diffusion pipeline, quantized UNet, and default sampler settings
    cfg, pipe, qnn, steps, guide = build_qnn(args.config, args.ckpt, fp16=True)

    # Prepares a mapping from weight bitwidth to its YAML config file
    wbits_to_yaml = {}
    for wb in args.wbits:
        candidate = Path(args.weight_dir, f"weight_{wb}.00.yaml")
        if candidate.exists():
            wbits_to_yaml[wb] = candidate
        else:
            alt = Path(args.weight_dir, f"weight_{wb:.2f}.yaml".replace(",", "."))
            if alt.exists():
                wbits_to_yaml[wb] = alt
            else:
                print(f"[WARN] no weight yaml for wbits={wb} in {args.weight_dir}")

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wbits", "res", "p50_ms", "p95_ms", "peak_mem_GB", "clipscore"])

        # Optional FP16 baseline (treated as wbits=16)
        try:
            cfg_fp16 = OmegaConf.load(args.config)
            _, pipe_fp16 = get_model(cfg_fp16.model, fp16=True, return_pipe=True)
            fp16_steps = int(cfg_fp16.calib_data.n_steps)
            fp16_guide = float(cfg_fp16.calib_data.scale_value)
            pipe_fp16.to("cuda")
        except Exception as e:
            print(f"[FP16][WARN] failed to load FP16 baseline from {args.config}: {e}")
            pipe_fp16 = None

        if pipe_fp16 is not None:
            wbits_fp16 = 16
            print(f"\n[SWEEP] FP16 baseline (wbits={wbits_fp16})")
            for res in args.res:
                # Runs a warmup pass for this resolution to populate caches
                torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.inference_mode():
                        _ = pipe_fp16(
                            prompt=[args.prompts[0]],
                            height=res,
                            width=res,
                            num_inference_steps=fp16_steps,
                            guidance_scale=fp16_guide,
                        ).images[0]
                except Exception as e:
                    print(f"[FP16][WARN] warmup failed res={res}: {e}")
                    writer.writerow([wbits_fp16, res, "nan", "nan", "nan", "nan"])
                    f.flush()
                    continue

                lat_ms, imgs = [], []
                for p in args.prompts:
                    try:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.inference_mode():
                            img = pipe_fp16(
                                prompt=[p],
                                height=res,
                                width=res,
                                num_inference_steps=fp16_steps,
                                guidance_scale=fp16_guide,
                            ).images[0]
                        torch.cuda.synchronize()
                        dt = (time.perf_counter() - t0) * 1000.0
                        lat_ms.append(dt)
                        imgs.append(img)
                    except Exception as e:
                        print(f"[FP16][WARN] generation failed res={res} prompt='{p}': {e}")
                        continue

                if not lat_ms:
                    print(f"[FP16][WARN] no successful samples for res={res}")
                    writer.writerow([wbits_fp16, res, "nan", "nan", "nan", "nan"])
                    f.flush()
                    continue

                p50 = float(np.percentile(lat_ms, 50))
                p95 = float(np.percentile(lat_ms, 95))
                peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(
                    f"[FP16][GEN] res={res} generated {len(imgs)} images; "
                    f"peak VRAM={peak_gb:.2f} GB"
                )

                cs = clipscore(args.prompts[: len(imgs)], imgs) if imgs else float("nan")

                writer.writerow(
                    [
                        wbits_fp16,
                        res,
                        f"{p50:.1f}" if np.isfinite(p50) else "nan",
                        f"{p95:.1f}" if np.isfinite(p95) else "nan",
                        f"{peak_gb:.3f}" if np.isfinite(peak_gb) else "nan",
                        f"{cs:.4f}" if np.isfinite(cs) else "nan",
                    ]
                )
                f.flush()
                torch.cuda.empty_cache()

        # Sweeps all configured quantized bitwidths
        for wbits, weight_yaml in wbits_to_yaml.items():
            print(f"\n[SWEEP] wbits={wbits} -> {weight_yaml}")
            apply_weight_bits(qnn, str(weight_yaml))
            pipe.unet = qnn
            pipe.to("cuda")

            for res in args.res:
                # Warmup to trigger compilation and caching for this config
                torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.inference_mode():
                        _ = pipe(
                            prompt=[args.prompts[0]],
                            height=res,
                            width=res,
                            num_inference_steps=steps,
                            guidance_scale=guide,
                        ).images[0]
                except Exception as e:
                    print(f"[WARN] warmup failed wbits={wbits} res={res}: {e}")
                    writer.writerow([wbits, res, "nan", "nan", "nan", "nan"])
                    f.flush()
                    continue

                lat_ms, imgs = [], []
                for p in args.prompts:
                    try:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.inference_mode():
                            img = pipe(
                                prompt=[p],
                                height=res,
                                width=res,
                                num_inference_steps=steps,
                                guidance_scale=guide,
                            ).images[0]
                        torch.cuda.synchronize()
                        lat_ms.append((time.perf_counter() - t0) * 1000.0)
                        imgs.append(img)
                    except Exception as e:
                        print(
                            f"[WARN] generation failed wbits={wbits} res={res} "
                            f"prompt='{p}': {e}"
                        )

                # Computes latency statistics for this configuration
                if lat_ms:
                    lat_ms.sort()
                    p50 = lat_ms[int(0.50 * (len(lat_ms) - 1))]
                    p95 = lat_ms[int(0.95 * (len(lat_ms) - 1))]
                else:
                    p50 = p95 = float("nan")

                peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(
                    f"[GEN] wbits={wbits} res={res} generated {len(imgs)} images; "
                    f"peak VRAM={peak_gb:.2f} GB"
                )

                # Computes CLIP score on the successful outputs
                cs = clipscore(args.prompts[: len(imgs)], imgs) if imgs else float("nan")

                writer.writerow(
                    [
                        wbits,
                        res,
                        f"{p50:.1f}" if np.isfinite(p50) else "nan",
                        f"{p95:.1f}" if np.isfinite(p95) else "nan",
                        f"{peak_gb:.3f}" if np.isfinite(peak_gb) else "nan",
                        f"{cs:.4f}" if np.isfinite(cs) else "nan",
                    ]
                )
                f.flush()

                print(
                    f"wbits={wbits} res={res} -> p50={p50:.1f} p95={p95:.1f} "
                    f"VRAM={peak_gb:.2f} GB  clip={cs if np.isfinite(cs) else 'nan'}"
                )

                torch.cuda.empty_cache()

    print(f"\n[OK] results written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
