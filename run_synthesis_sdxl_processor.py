import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf
from diffusers import DDIMScheduler, StableDiffusionPipeline, DiffusionPipeline
from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

from masactrl.masactrl_processor import register_attention_processor

torch.cuda.set_device(0)  # set the GPU device

# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
weight_dtype = torch.float16
model_path = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False
)
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    scheduler=scheduler,
    torch_dtype=weight_dtype
).to(device)


def consistent_synthesis():
    seed = 42
    seed_everything(seed)

    out_dir_ori = "./workdir/masactrl_exp/oldman_smiling"
    os.makedirs(out_dir_ori, exist_ok=True)

    prompts = [
        "A portrait of an old man, facing camera, best quality",
        "A portrait of an old man, facing camera, smiling, best quality",
    ]

    # inference the synthesized image with MasaCtrl
    # TODO: note that the hyper paramerter of MasaCtrl for SDXL may be not optimal
    STEP = 4
    LAYER_LIST = [44, 54, 64]  # run the synthesis with MasaCtrl at three different layer configs
    MODEL_TYPE = "SDXL"

    # initialize the noise map
    start_code = torch.randn([1, 4, 128, 128], dtype=weight_dtype, device=device)
    # start_code = None
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # inference the synthesized image without MasaCtrl
    image_ori = pipe(prompts, latents=start_code, guidance_scale=7.5).images

    for LAYER in LAYER_LIST:
        # hijack the attention module with MasaCtrl processor
        processor_args = {
            "start_step": STEP,
            "start_layer": LAYER,
            "model_type": MODEL_TYPE
        }
        register_attention_processor(pipe.unet, processor_type="MasaCtrlProcessor")

        # inference the synthesized image
        image_masactrl = pipe(prompts, latents=start_code, guidance_scale=7.5).images

        sample_count = len(os.listdir(out_dir_ori))
        out_dir = os.path.join(out_dir_ori, f"sample_{sample_count}")
        os.makedirs(out_dir, exist_ok=True)
        image_ori[0].save(os.path.join(out_dir, f"source_step{STEP}_layer{LAYER}.png"))
        image_ori[1].save(os.path.join(out_dir, f"without_step{STEP}_layer{LAYER}.png"))
        image_masactrl[-1].save(os.path.join(out_dir, f"masactrl_step{STEP}_layer{LAYER}.png"))
        with open(os.path.join(out_dir, f"prompts.txt"), "w") as f:
            for p in prompts:
                f.write(p + "\n")
            f.write(f"seed: {seed}\n")
        print("Syntheiszed images are saved in", out_dir)


if __name__ == "__main__":
    consistent_synthesis()
