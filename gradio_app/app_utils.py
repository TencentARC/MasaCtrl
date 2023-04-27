import gradio as gr
import numpy as np
import torch
from diffusers import DDIMScheduler
from pytorch_lightning import seed_everything

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import (AttentionBase,
                                     regiter_attention_editor_diffusers)


torch.set_grad_enabled(False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
model_path = "andite/anything-v4.0"
scheduler = DDIMScheduler(beta_start=0.00085,
                          beta_end=0.012,
                          beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
model = MasaCtrlPipeline.from_pretrained(model_path,
                                         scheduler=scheduler).to(device)

global_context = {
    "model_path": model_path,
    "scheduler": scheduler,
    "model": model,
    "device": device
}