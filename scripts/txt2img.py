import argparse, os, datetime, gc, yaml
import logging
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch.cuda import amp
from contextlib import nullcontext
import json
import sys

from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_model, load_quant_params, prepare_coco_text_and_image
from controller import ResolutionController

from tqdm.auto import tqdm
import sys

logger = logging.getLogger(__name__)
# Uses a resolution controller to adapt image resolution based on an external queue length
rc = ResolutionController(q_low=1, q_high=4, default=1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="how many batches to produce for each given prompt. A.k.a. batch size",
    )

    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to config which constructs model, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="path for generated images, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--num_imgs",
        type=int,
        default=32,
        help="the number of the output images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
    )
    # Optional externally provided queue length used by the resolution controller
    parser.add_argument(
        "--queue_len",
        type=int,
        default=None,
        help="External queue length for load-dependent control"
    )
    
    # ---- Overrides Turbo (facultatifs) ----
    parser.add_argument(
        "--override_steps", type=int, default=None,
        help="Force num_inference_steps (ex: 1 pour SDXL-Turbo)."
    )
    parser.add_argument(
        "--override_cfg", type=float, default=None,
        help="Force guidance_scale (ex: 0.0 pour SDXL-Turbo)."
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    opt.outdir = os.path.join(opt.base_path, 'generated_images')
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    log_path = os.path.join(opt.base_path, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # load the config from the log path
    if opt.config is None:
        opt.config = os.path.join(opt.base_path, 'config.yaml')
    if opt.image_folder is None:
        opt.image_folder = os.path.join(opt.base_path, 'generated_images')
    config = OmegaConf.load(f"{opt.config}")
    
    # Prints which config file and pretrained model identifier are being used
    print("CONFIG FROM:", opt.config)
    print("MODEL ID   :", config.model.get("pretrained_model_name_or_path", "N/A"))

    if opt.cfg is None:
        opt.cfg = config.calib_data.scale_value

    # adapter_id = getattr(config.model, "adapter_id", None)
    # adapter_cache_dir = getattr(config.model, "adapter_cache_dir", None)
    model, pipe = get_model(config.model, fp16=opt.fp16, return_pipe=True)
    # Tries to install an LCM scheduler, which is well suited for Turbo-style inference
    try:
        from diffusers import LCMScheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        logger.info("Using LCM scheduler for Turbo.")
    except Exception as e:
        logger.warning(f"LCM scheduler not set: {e}")
    ####
    num_timesteps = config.calib_data.n_steps
    # valeurs réellement utilisées
    opt._steps = opt.override_steps if opt.override_steps is not None else num_timesteps
    opt._cfg   = opt.override_cfg   if opt.override_cfg   is not None else opt.cfg

    # ---- Valeurs Turbo par défaut (modifiables au CLI via --override_*) ----
    default_steps = 1      # SDXL-Turbo: 1 step
    default_cfg   = 0.0    # SDXL-Turbo: guidance ~ 0

    opt._steps = int(opt.override_steps) if opt.override_steps is not None else default_steps
    opt._cfg   = float(opt.override_cfg) if opt.override_cfg is not None else default_cfg

    assert (config.conditional)
    model.cuda()
    model.eval()
    logger.info(model)

    dtype = torch.float32 if not opt.fp16 else torch.float16
    if opt.fp16:
        model = model.half()  # make some newly genrated quant-related modules FP16

    # Forward
    if opt.prompt is None:
        json_file = "./scripts/utils/captions_val2014.json"
        prompt_list, image_path = prepare_coco_text_and_image(json_file=json_file)
        prompts = prompt_list[0:opt.num_imgs]
    else:
        prompts = [opt.prompt] * opt.num_imgs

    # Uses opt._steps so the generator follows Turbo defaults or CLI overrides
    gen_image(prompts, model, pipe, opt._steps, opt)


def gen_image(prompt, unet, pipe, num_timesteps, opt):
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.manual_seed(opt.seed)  # Seed generator to create the initial latent noise
    total = len(prompt)
    batch_size = opt.batch_size
    assert (total >= batch_size), "the length of prompts should larger than batch_size"
    num = total // batch_size

    img_id = 0
    logger.info(f"starting from image {img_id}")

    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    pipe.unet = unet.half() if opt.fp16 else unet

    pipe.to("cuda")
    with torch.no_grad():
        for i in tqdm(
            range(num), desc="Generating image samples for FID evaluation."
        ):
            with amp.autocast(enabled=False):
                # Derives an effective queue length and asks the controller for a target resolution
                queue_len = 0 if opt.queue_len is None else max(0, opt.queue_len)
                r = rc.choose(queue_len)  # selects among 512, 768, 1024
                image = pipe(
                    prompt=prompt[img_id:img_id + batch_size],
                    height=r,
                    width=r,
                    num_inference_steps=num_timesteps,   # receives opt._steps from main
                    guidance_scale=opt._cfg              # uses Turbo CFG or CLI override
                ).images

            for j in range(batch_size):
                # Uses os.path.join to build the output path in a portable way
                image[j].save(os.path.join(opt.image_folder, f"{img_id}.png"))
                img_id += 1


if __name__ == "__main__":
    main()
