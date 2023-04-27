import os
import numpy as np
import gradio as gr
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from torchvision.io import read_image
from pytorch_lightning import seed_everything

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import (AttentionBase,
                                     regiter_attention_editor_diffusers)

from .app_utils import global_context

torch.set_grad_enabled(False)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
#     "cpu")

# model_path = "CompVis/stable-diffusion-v1-4"
# scheduler = DDIMScheduler(beta_start=0.00085,
#                           beta_end=0.012,
#                           beta_schedule="scaled_linear",
#                           clip_sample=False,
#                           set_alpha_to_one=False)
# model = MasaCtrlPipeline.from_pretrained(model_path,
#                                          scheduler=scheduler).to(device)


def load_image(image_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)


def real_image_editing(source_image, target_prompt,
                       starting_step, starting_layer, ddim_steps, scale, seed,
                       appended_prompt, negative_prompt):
    from masactrl.masactrl import MutualSelfAttentionControl

    model = global_context["model"]
    device = global_context["device"]

    seed_everything(seed)

    with torch.no_grad():
        if appended_prompt is not None:
            target_prompt += appended_prompt
        ref_prompt = ""
        prompts = [ref_prompt, target_prompt]

        # invert the image into noise map
        if isinstance(source_image, np.ndarray):
            source_image = torch.from_numpy(source_image).to(device) / 127.5 - 1.
            source_image = source_image.unsqueeze(0).permute(0, 3, 1, 2)
            source_image = F.interpolate(source_image, (512, 512))

        start_code, latents_list = model.invert(source_image,
                                                ref_prompt,
                                                guidance_scale=scale,
                                                num_inference_steps=ddim_steps,
                                                return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1)

        # recontruct the image with inverted DDIM noise map
        editor = AttentionBase()
        regiter_attention_editor_diffusers(model, editor)
        image_fixed = model([target_prompt],
                            latents=start_code[-1:],
                            num_inference_steps=ddim_steps,
                            guidance_scale=scale)
        image_fixed = image_fixed.cpu().permute(0, 2, 3, 1).numpy()

        # inference the synthesized image with MasaCtrl
        # hijack the attention module
        controller = MutualSelfAttentionControl(starting_step, starting_layer)
        regiter_attention_editor_diffusers(model, controller)

        # inference the synthesized image
        image_masactrl = model(prompts,
                               latents=start_code,
                               guidance_scale=scale)
        image_masactrl = image_masactrl.cpu().permute(0, 2, 3, 1).numpy()

    return [
        image_masactrl[0],
        image_fixed[0],
        image_masactrl[1]
    ]  # source, fixed seed, masactrl


def create_demo_editing():
    with gr.Blocks() as demo:
        gr.Markdown("## **Input Settings**")
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(label="Source Image", value=os.path.join(os.path.dirname(__file__), "images/corgi.jpg"), interactive=True)
                target_prompt = gr.Textbox(label="Target Prompt",
                                        value='A photo of a running corgi',
                                        interactive=True)
                with gr.Row():
                    ddim_steps = gr.Slider(label="DDIM Steps",
                                        minimum=1,
                                        maximum=999,
                                        value=50,
                                        step=1)
                    starting_step = gr.Slider(label="Step of MasaCtrl",
                                            minimum=0,
                                            maximum=999,
                                            value=4,
                                            step=1)
                    starting_layer = gr.Slider(label="Layer of MasaCtrl",
                                            minimum=0,
                                            maximum=16,
                                            value=10,
                                            step=1)
                run_btn = gr.Button(label="Run")
            with gr.Column():
                appended_prompt = gr.Textbox(label="Appended Prompt", value='')
                negative_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Row():
                    scale = gr.Slider(label="CFG Scale",
                                    minimum=0.1,
                                    maximum=30.0,
                                    value=7.5,
                                    step=0.1)
                    seed = gr.Slider(label="Seed",
                                    minimum=-1,
                                    maximum=2147483647,
                                    value=42,
                                    step=1)

        gr.Markdown("## **Output**")
        with gr.Row():
            image_recons = gr.Image(label="Source Image")
            image_fixed = gr.Image(label="Image with Fixed Seed")
            image_masactrl = gr.Image(label="Image with MasaCtrl")

        inputs = [
            source_image, target_prompt, starting_step, starting_layer, ddim_steps,
            scale, seed, appended_prompt, negative_prompt
        ]
        run_btn.click(real_image_editing, inputs,
                    [image_recons, image_fixed, image_masactrl])

        gr.Examples(
            [[os.path.join(os.path.dirname(__file__), "images/corgi.jpg"),
              "A photo of a running corgi"],
            [os.path.join(os.path.dirname(__file__), "images/person.png"),
             "A photo of a person, black t-shirt, raising hand"],
            ],
            [source_image, target_prompt]
        )
    return demo


if __name__ == "__main__":
    demo_editing = create_demo_editing()
    demo_editing.launch()
