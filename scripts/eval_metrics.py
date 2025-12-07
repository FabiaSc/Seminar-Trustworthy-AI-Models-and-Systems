#!/usr/bin/env python3
import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image

import torchvision.transforms as T
import torchvision.models as tv_models

# Try scipy for sqrtm in FID. If not available, we'll skip FID.
try:
    import numpy as np
    import scipy.linalg
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    import numpy as np


# ---------------------------
# COCO caption / image loader
# ---------------------------

def load_coco_pairs(
    coco_json_path: str,
    coco_img_root: str,
    max_items: int,
) -> List[Dict]:
    """
    Returns a list of dicts:
    [
      {
        "caption": "a group of people ...",
        "coco_image_path": "scripts/utils/val2014/COCO_val2014_000000391895.jpg",
        "image_id": 391895
      },
      ...
    ]

    We walk through the COCO "annotations" and map image_id -> file_name using
    the "images" section. We then stop after max_items.
    """
    with open(coco_json_path, "r") as f:
        coco_info = json.load(f)

    # Map image_id -> file_name so we can resolve full paths
    id_to_fname = {}
    for img_entry in coco_info.get("images", []):
        img_id = img_entry["id"]
        fname = img_entry["file_name"]
        id_to_fname[img_id] = fname

    pairs = []
    for ann in coco_info.get("annotations", []):
        img_id = ann["image_id"]
        caption = ann["caption"]
        if img_id not in id_to_fname:
            continue
        fname = id_to_fname[img_id]
        img_path = os.path.join(coco_img_root, fname)
        pairs.append(
            {
                "caption": caption,
                "coco_image_path": img_path,
                "image_id": img_id,
            }
        )
        if len(pairs) >= max_items:
            break

    return pairs


# ---------------------------
# Generated image listing
# ---------------------------

def load_generated_images(gen_dir: str, max_items: int) -> List[str]:
    """
    Returns a sorted list of paths to generated images, truncated to max_items.
    We assume files like 0.png, 1.png, etc.
    """
    all_files = []
    for fname in os.listdir(gen_dir):
        fpath = os.path.join(gen_dir, fname)
        if os.path.isfile(fpath):
            low = fname.lower()
            if low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg"):
                all_files.append(fpath)

    # Sort numerically if possible (so 0.png,1.png,... aligns with prompts order)
    def _key(x):
        stem = os.path.splitext(os.path.basename(x))[0]
        try:
            return int(stem)
        except ValueError:
            return stem

    all_files = sorted(all_files, key=_key)
    if len(all_files) > max_items:
        all_files = all_files[:max_items]
    return all_files


# ---------------------------
# Inception (FID)
# ---------------------------

def get_inception_embedder(device: str):
    """
    Build an InceptionV3 feature extractor for FID.
    We DO NOT pass aux_logits=False (that caused your crash with the newer torchvision).
    We then replace the final fc layer with Identity() to get 2048-d pool features.
    """
    inception = tv_models.inception_v3(pretrained=True)
    inception.fc = torch.nn.Identity()
    inception.to(device)
    inception.eval()

    # Standard InceptionV3 preprocessing
    incept_preproc = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return inception, incept_preproc


@torch.no_grad()
def inception_features(
    model,
    preproc,
    img_paths: List[str],
    device: str,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Run a list of image paths through InceptionV3 (fc replaced w/ Identity).
    Returns tensor of shape [N, 2048].
    """
    feats = []
    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start:start + batch_size]
        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = preproc(img)
            batch_imgs.append(img)
        batch_tensor = torch.stack(batch_imgs, dim=0).to(device)
        out = model(batch_tensor)

        # inception_v3 may return InceptionOutputs(logits, aux), depending on version
        if isinstance(out, tuple) or hasattr(out, "logits"):
            # torchscript style or InceptionOutputs
            # try grabbing first element / logits
            if hasattr(out, "logits"):
                out_main = out.logits
            else:
                out_main = out[0]
        else:
            out_main = out

        feats.append(out_main.detach().cpu())
    feats = torch.cat(feats, dim=0)
    return feats  # [N, 2048]


def compute_fid(feats_real: torch.Tensor, feats_fake: torch.Tensor) -> float:
    """
    Standard FID:
    ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2 * sqrt(Sigma_r Sigma_f))

    Uses scipy.linalg.sqrtm if available, otherwise falls back to
    eigen-decomposition approximation. If neither works, returns NaN.
    """
    feats_real_np = feats_real.numpy()
    feats_fake_np = feats_fake.numpy()

    mu_r = np.mean(feats_real_np, axis=0)
    mu_f = np.mean(feats_fake_np, axis=0)

    sigma_r = np.cov(feats_real_np, rowvar=False)
    sigma_f = np.cov(feats_fake_np, rowvar=False)

    diff = mu_r - mu_f
    diff_sq = diff.dot(diff)

    if HAS_SCIPY:
        covmean = scipy.linalg.sqrtm(sigma_r.dot(sigma_f))
        # sqrtm can return complex due to numeric issues; take real part
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    else:
        # Fallback: symmetric approximation
        # This is not exact if sigma_r and sigma_f don't commute,
        # but it's a decent fallback so you at least get *some* number.
        # We symmetrize the product.
        prod = sigma_r.dot(sigma_f)
        prod = (prod + prod.T) / 2.0
        w, v = np.linalg.eigh(prod)
        w_clip = np.clip(w, a_min=0, a_max=None)
        covmean = (v * np.sqrt(w_clip)) @ v.T

    trace_term = (
        np.trace(sigma_r)
        + np.trace(sigma_f)
        - 2.0 * np.trace(covmean)
    )

    fid = float(diff_sq + trace_term)
    return fid


# ---------------------------
# CLIP + IR metrics
# ---------------------------

def get_clip_model(device: str):
    """
    Load CLIP from Hugging Face transformers.
    We'll use CLIPModel + CLIPProcessor for ViT-L/14 style embeddings.
    """
    from transformers import CLIPModel, CLIPProcessor

    # Auto-picks an open CLIP checkpoint like "openai/clip-vit-large-patch14".
    # You can change this string if the repo used a different CLIP backbone.
    clip_name = "openai/clip-vit-large-patch14"

    model = CLIPModel.from_pretrained(clip_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_name)

    model.eval()
    return model, processor


@torch.no_grad()
def compute_clip_and_ir(
    clip_model,
    clip_processor,
    gen_img_paths: List[str],
    captions: List[str],
    device: str,
    batch_size: int = 16,
) -> Tuple[float, float]:
    """
    Returns:
      (mean_clip_score, ir_at_1)

    mean_clip_score:
        average cosine similarity(image_emb, text_emb) for the matching pair

    ir_at_1:
        For each image, does its correct caption rank highest among all given captions?
        (basically retrieval accuracy@1 within this eval set)
    """
    # We'll embed images in batches, and texts in batches, then combine.

    # 1. Embed all images
    img_embeds_all = []
    for start in range(0, len(gen_img_paths), batch_size):
        batch_paths = gen_img_paths[start:start + batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = clip_processor(
            images=batch_imgs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        img_out = clip_model.get_image_features(**{k: v for k, v in inputs.items() if k.startswith("pixel_values")})
        # L2-normalize
        img_out = F.normalize(img_out, p=2, dim=-1)
        img_embeds_all.append(img_out.cpu())
    img_embeds_all = torch.cat(img_embeds_all, dim=0)  # [N, D]

    # 2. Embed all captions
    text_embeds_all = []
    for start in range(0, len(captions), batch_size):
        batch_caps = captions[start:start + batch_size]
        inputs = clip_processor(
            text=batch_caps,
            return_tensors="pt",
            padding=True,
        ).to(device)
        txt_out = clip_model.get_text_features(**{k: v for k, v in inputs.items() if k.startswith("input_ids") or k.startswith("attention_mask")})
        txt_out = F.normalize(txt_out, p=2, dim=-1)
        text_embeds_all.append(txt_out.cpu())
    text_embeds_all = torch.cat(text_embeds_all, dim=0)  # [N, D]

    assert img_embeds_all.shape[0] == text_embeds_all.shape[0], \
        "We expect same number of images and captions for pairwise metrics."

    N = img_embeds_all.shape[0]

    # 3. CLIP score = diagonal cosine sim (image i vs caption i)
    sims_diag = torch.sum(img_embeds_all * text_embeds_all, dim=-1)  # [N]
    mean_clip = sims_diag.mean().item()

    # 4. IR@1 (image retrieval using text): 
    # For each image, find which caption is most similar.
    # Build full similarity matrix [N, N] = img_embeds_all @ text_embeds_all^T.
    sim_matrix = img_embeds_all @ text_embeds_all.t()  # [N, N]
    # For each row i (image i), find index of best-matching caption.
    top_idx = torch.argmax(sim_matrix, dim=1)  # [N]
    correct = (top_idx == torch.arange(N)).float()
    ir_at_1 = correct.mean().item()

    return mean_clip, ir_at_1


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="Folder of generated images to evaluate (FP16 or MixDQ output).",
    )
    parser.add_argument(
        "--num_imgs",
        type=int,
        default=64,
        help="How many images/captions to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        default="scripts/utils/captions_val2014.json",
        help="Path to COCO captions_val2014.json",
    )
    parser.add_argument(
        "--coco_root",
        type=str,
        default="scripts/utils/val2014",
        help="Path to COCO val2014/ image directory",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # 1. Load COCO captions + real images
    print("Loading COCO captions + real image paths...")
    coco_pairs = load_coco_pairs(
        coco_json_path=args.coco_json,
        coco_img_root=args.coco_root,
        max_items=args.num_imgs,
    )
    print(f"Loaded {len(coco_pairs)} prompt/image pairs from COCO.")

    # We will align each generated image index with that same index in coco_pairs.
    gt_image_paths = [p["coco_image_path"] for p in coco_pairs]
    gt_captions = [p["caption"] for p in coco_pairs]

    # 2. Load generated images
    print("Listing generated images...")
    gen_image_paths = load_generated_images(
        gen_dir=args.gen_dir,
        max_items=args.num_imgs,
    )
    print(f"Found {len(gen_image_paths)} generated images in {args.gen_dir}")

    if len(gen_image_paths) == 0:
        raise RuntimeError("No generated images found. Check --gen_dir path.")

    # 3. Build InceptionV3 for FID features
    print("Building InceptionV3 for FID features...")
    inception, incept_preproc = get_inception_embedder(device=device)

    with torch.no_grad():
        # Extract Inception features for real COCO imgs
        feats_real = inception_features(
            inception,
            incept_preproc,
            gt_image_paths,
            device=device,
            batch_size=16,
        )  # [N, 2048]

        # Extract Inception features for generated imgs
        feats_fake = inception_features(
            inception,
            incept_preproc,
            gen_image_paths,
            device=device,
            batch_size=16,
        )  # [N, 2048]

    if feats_real.shape[0] != feats_fake.shape[0]:
        print(
            f"WARNING: real({feats_real.shape[0]}) vs gen({feats_fake.shape[0]}) "
            f"mismatch; FID will still compute but may not be 1:1 comparable."
        )

    if HAS_SCIPY:
        fid_val = compute_fid(feats_real, feats_fake)
    else:
        print("WARNING: scipy not found, using fallback FID approximation.")
        fid_val = compute_fid(feats_real, feats_fake)

    # 4. CLIP / IR metrics
    print("Building CLIP model for CLIP score + IR...")
    clip_model, clip_processor = get_clip_model(device=device)

    mean_clip, ir_at_1 = compute_clip_and_ir(
        clip_model,
        clip_processor,
        gen_image_paths,
        gt_captions[: len(gen_image_paths)],
        device=device,
        batch_size=16,
    )

    # 5. Report
    print("======== RESULTS ========")
    print(f"FID:        {fid_val:.4f}")
    print(f"CLIP score: {mean_clip:.4f}")
    print(f"IR@1:       {ir_at_1:.4f}")
    print("=========================")


if __name__ == "__main__":
    main()
