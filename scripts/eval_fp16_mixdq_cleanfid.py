#!/usr/bin/env python
import argparse
import os
import glob

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from cleanfid import fid as cleanfid
import clip
import ImageReward as RM


def list_images(folder):
    # Collects image files in this folder with common extensions
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files)
    # Fails early if the folder does not contain any image
    if not files:
        raise ValueError(f"No images found in {folder}")
    return files


def load_prompts(path):
    # Ensures the prompts file exists before reading it
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")
    # Reads one prompt per non-empty line and strips whitespace
    prompts = [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]
    # Avoids silently running with an empty prompt list
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def compute_fid_cleanfid(ref_dir, gen_dir, device_str="cuda"):
    # Delegates FID computation to clean-fid for the two image folders
    print(f"\n[FID] clean-fid between\n  ref = {ref_dir}\n  gen = {gen_dir}")
    score = cleanfid.compute_fid(
        ref_dir,
        gen_dir,
        device=device_str,
        num_workers=8,
    )
    return float(score)


def compute_clip_scores(image_paths, prompts, model, preprocess, device, batch_size=32):
    # Enforces a one-to-one mapping between images and prompts
    assert len(image_paths) == len(prompts), \
        f"#images ({len(image_paths)}) != #prompts ({len(prompts)})"

    print(f"\n[CLIP] {len(image_paths)} samples")
    sims = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]

        # Loads and preprocesses the image batch for CLIP
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        img_tensor = torch.stack([preprocess(img) for img in images]).to(device)
        # Tokenizes the corresponding prompts
        txt_tokens = clip.tokenize(batch_prompts).to(device)

        with torch.no_grad():
            # Encodes images and texts into CLIP feature space
            img_feat = model.encode_image(img_tensor).float()
            txt_feat = model.encode_text(txt_tokens).float()

        # l2-normalizes features so the dot product becomes cosine similarity (CLIP score)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        batch_sim = (img_feat * txt_feat).sum(dim=-1)
        sims.append(batch_sim.cpu().numpy())

    # Aggregates all similarities and returns mean and std for reporting
    sims = np.concatenate(sims, axis=0)
    return float(sims.mean()), float(sims.std())


def compute_image_reward(image_paths, prompts, irmodel, batch_size=16):
    # Again enforces a one-to-one mapping between images and prompts
    assert len(image_paths) == len(prompts), \
        f"#images ({len(image_paths)}) != #prompts ({len(prompts)})"

    print(f"\n[ImageReward] {len(image_paths)} samples")
    scores = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_prompts = prompts[i:i + batch_size]
        for prompt, img_path in zip(batch_prompts, batch_paths):
            with torch.no_grad():
                # Queries ImageReward with the text prompt and image path
                s = irmodel.score(prompt, img_path)
            scores.append(float(s))

    # Summarizes ImageReward scores with mean and std
    scores = np.asarray(scores, dtype=np.float32)
    return float(scores.mean()), float(scores.std())


def main():
    # Defines CLI arguments for folders, prompts, and evaluation settings
    parser = argparse.ArgumentParser(
        description="Evaluate FP16 vs MixDQ on FID (cleanfid), CLIP, and ImageReward."
    )
    parser.add_argument("--ref_folder", type=str, required=True)
    parser.add_argument("--fp16_folder", type=str, required=True)
    parser.add_argument("--mixdq_folder", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Ensures all required directories exist before running expensive work
    for d in [args.ref_folder, args.fp16_folder, args.mixdq_folder]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    ref_imgs   = list_images(args.ref_folder)
    fp16_imgs  = list_images(args.fp16_folder)
    mixdq_imgs = list_images(args.mixdq_folder)
    prompts    = load_prompts(args.prompts_file)

    # Aligns the number of samples with the available prompts
    n = min(len(fp16_imgs), len(mixdq_imgs), len(prompts))
    if args.num_samples is not None:
        n = min(n, args.num_samples)

    fp16_imgs  = fp16_imgs[:n]
    mixdq_imgs = mixdq_imgs[:n]
    prompts    = prompts[:n]

    print("-------------------------------------------------")
    print(f"#ref_images    (for FID) : {len(ref_imgs)}")
    print(f"#fp16_images            : {len(fp16_imgs)}")
    print(f"#mixdq_images           : {len(mixdq_imgs)}")
    print(f"#prompts                : {len(prompts)}")
    print("-------------------------------------------------")

    # Picks the evaluation device, preferring user input, then CUDA if available
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Loads the CLIP and ImageReward models once for all metrics
    print("\n[Init] Loading CLIP ViT-L/14...")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

    print("[Init] Loading ImageReward-v1.0...")
    irmodel = RM.load("ImageReward-v1.0")
    if hasattr(irmodel, "device"):
        # Moves ImageReward to the same device when the API allows it
        irmodel.device = device

    # Computes FID between reference images and each variant
    fid_fp16  = compute_fid_cleanfid(args.ref_folder, args.fp16_folder,  device_str=dev_str)
    fid_mixdq = compute_fid_cleanfid(args.ref_folder, args.mixdq_folder, device_str=dev_str)

    # Computes CLIP alignment for FP16 and MixDQ samples
    clip_mean_fp16, clip_std_fp16 = compute_clip_scores(
        fp16_imgs, prompts, clip_model, clip_preprocess, device,
        batch_size=args.batch_size,
    )
    clip_mean_mixdq, clip_std_mixdq = compute_clip_scores(
        mixdq_imgs, prompts, clip_model, clip_preprocess, device,
        batch_size=args.batch_size,
    )

    # Computes ImageReward scores for both variants
    ir_mean_fp16, ir_std_fp16 = compute_image_reward(
        fp16_imgs, prompts, irmodel,
        batch_size=args.batch_size,
    )
    ir_mean_mixdq, ir_std_mixdq = compute_image_reward(
        mixdq_imgs, prompts, irmodel,
        batch_size=args.batch_size,
    )

    # Prints a compact summary comparing FP16 and MixDQ across all metrics
    print("\n======================================")
    print("           FP16 vs MixDQ EVALUATION   ")
    print("======================================")
    print("FID (clean-fid, lower is better):")
    print(f"  FP16  : {fid_fp16:.4f}")
    print(f"  MixDQ : {fid_mixdq:.4f}")
    print("--------------------------------------")
    print("CLIP (cosine sim, higher is better):")
    print(f"  FP16  : mean={clip_mean_fp16:.4f}, std={clip_std_fp16:.4f}")
    print(f"  MixDQ : mean={clip_mean_mixdq:.4f}, std={clip_std_mixdq:.4f}")
    print("--------------------------------------")
    print("ImageReward (higher is better):")
    print(f"  FP16  : mean={ir_mean_fp16:.4f}, std={ir_std_fp16:.4f}")
    print(f"  MixDQ : mean={ir_mean_mixdq:.4f}, std={ir_std_mixdq:.4f}")
    print("======================================\n")


if __name__ == "__main__":
    main()
