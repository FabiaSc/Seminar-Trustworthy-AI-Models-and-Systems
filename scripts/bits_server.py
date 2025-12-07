# scripts/serve_bits.py
import io, base64, time, os, yaml
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
from omegaconf import OmegaConf

# MixDQ utils
from qdiff.utils import get_model, load_quant_params
from qdiff.models.quant_model import QuantModel

# Exposes a small FastAPI app to serve MixDQ generations at different bitwidths
app = FastAPI(title="MixDQ Live Bits")
DTYPE = torch.float16

# Defines config and checkpoint paths for the quantized SDXL Turbo setup
CONFIG = "./configs/stable-diffusion/sdxl_turbo.yaml"
CKPT   = "./logs/sdxl_mixdq_eval/ckpt.pth"
WEIGHT_DIR = "./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/weight"
# Preloads weight quant configs for the supported bitwidths
WBITS_CFG = {
    8: yaml.safe_load(open(os.path.join(WEIGHT_DIR, "weight_8.00.yaml"))),
    6: yaml.safe_load(open(os.path.join(WEIGHT_DIR, "weight_6.00.yaml"))),
    4: yaml.safe_load(open(os.path.join(WEIGHT_DIR, "weight_4.00.yaml"))),
}

def build_qnn(config_yaml: str, ckpt_path: str, fp16=True):
    # Loads the model config and constructs a quant ready model and its diffusion pipeline
    cfg = OmegaConf.load(config_yaml)
    model, pipe = get_model(cfg.model, fp16=fp16, return_pipe=True, convert_model_for_quant=True)
    # Wraps the model in QuantModel to attach weight and activation quantization modules
    qnn = QuantModel(
        model=model,
        weight_quant_params=cfg.quant.weight.quantizer,
        act_quant_params=cfg.quant.activation.quantizer,
    )
    qnn.cuda().eval()
    if fp16:
        qnn = qnn.half()
    # Enables weight quantization only (A16) and marks both weight and activation init as done
    qnn.set_quant_state(True, False)
    qnn.set_quant_init_done("weight")
    qnn.set_quant_init_done("activation")
    # Loads learned quant parameters from the checkpoint
    load_quant_params(qnn, ckpt_path, dtype=DTYPE)
    # Reuses calibration steps and guidance scale as defaults for inference
    steps = cfg.calib_data.n_steps
    guidance = cfg.calib_data.scale_value
    return cfg, pipe, qnn, steps, guidance

# Builds the quantized network and pipeline once at startup
CFG, PIPE, QNN, DEFAULT_STEPS, DEFAULT_GUIDE = build_qnn(CONFIG, CKPT, fp16=True)

class GenRequest(BaseModel):
    # Describes the request schema for image generation
    prompt: str
    res: int = 1024         # 512/768/1024
    wbits: int = 8          # 4/6/8
    steps: int | None = None
    cfg_scale: float | None = None

def pil_to_b64(img: Image.Image) -> str:
    # Encodes a PIL image as base64 PNG so it can be returned in JSON
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.get("/health")
def health():
    # Exposes a simple health check endpoint for monitoring
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenRequest):
    # Switches the quantized model weights to the requested bitwidth (activations stay in fp16)
    QNN.load_bitwidth_config(model=QNN, bit_config=WBITS_CFG[req.wbits], bit_type="weight")
    PIPE.unet = QNN
    PIPE.to("cuda")

    # Falls back to config defaults if the client omits steps or cfg scale
    steps = req.steps or DEFAULT_STEPS
    cfg_sc = req.cfg_scale or DEFAULT_GUIDE

    # Tracks latency and peak GPU memory for this request
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    img = PIPE(
        prompt=[req.prompt],
        height=req.res,
        width=req.res,
        num_inference_steps=steps,
        guidance_scale=cfg_sc,
    ).images[0]
    dt_ms = (time.perf_counter() - t0) * 1000.0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Returns latency, memory usage, and the generated image encoded as base64 PNG
    return {
        "lat_ms": round(dt_ms, 1),
        "peak_mem_gb": round(peak_gb, 3),
        "res": req.res,
        "wbits": req.wbits,
        "image_png_base64": pil_to_b64(img),
    }
