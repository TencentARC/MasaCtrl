import gradio as gr
import numpy as np
import torch
from diffusers import DDIMScheduler
from pytorch_lightning import seed_everything

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import (AttentionBase,
                                     regiter_attention_editor_diffusers)

from .app_utils import global_context

torch.set_grad_enabled(False)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
#     "cpu")
# model_path = "andite/anything-v4.0"
# scheduler = DDIMScheduler(beta_start=0.00085,
#                           beta_end=0.012,
#                           beta_schedule="scaled_linear",
#                           clip_sample=False,
#                           set_alpha_to_one=False)
# model = MasaCtrlPipeline.from_pretrained(model_path,
#                                          scheduler=scheduler).to(device)


def consistent_synthesis(source_prompt, target_prompt, starting_step,
                         starting_layer, image_resolution, ddim_steps, scale,
                         seed, appended_prompt, negative_prompt):
    from masactrl.masactrl import MutualSelfAttentionControl

    model = global_context["model"]
    device = global_context["device"]

    seed_everything(seed)

    with torch.no_grad():
        if appended_prompt is not None:
            source_prompt += appended_prompt
            target_prompt += appended_prompt
        prompts = [source_prompt, target_prompt]

        # initialize the noise map
        start_code = torch.randn([1, 4, 64, 64], device=device)
        start_code = start_code.expand(len(prompts), -1, -1, -1)

        # inference the synthesized image without MasaCtrl
        editor = AttentionBase()
        regiter_attention_editor_diffusers(model, editor)
        target_image_ori = model([target_prompt],
                                 latents=start_code[-1:],
                                 guidance_scale=7.5)
        target_image_ori = target_image_ori.cpu().permute(0, 2, 3, 1).numpy()

        # inference the synthesized image with MasaCtrl
        # hijack the attention module
        controller = MutualSelfAttentionControl(starting_step, starting_layer)
        regiter_attention_editor_diffusers(model, controller)

        # inference the synthesized image
        image_masactrl = model(prompts, latents=start_code, guidance_scale=7.5)
        image_masactrl = image_masactrl.cpu().permute(0, 2, 3, 1).numpy()

    return [image_masactrl[0], target_image_ori[0],
            image_masactrl[1]]  # source, fixed seed, masactrl


def create_demo_synthesis():
    with gr.Blocks() as demo:
        gr.Markdown("## **Input Settings**")
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(
                    label="Source Prompt",
                    value='1boy, casual, outdoors, sitting',
                    interactive=True)
                target_prompt = gr.Textbox(
                    label="Target Prompt",
                    value='1boy, casual, outdoors, standing',
                    interactive=True)
                with gr.Row():
                    ddim_steps = gr.Slider(label="DDIM Steps",
                                            minimum=1,
                                            maximum=999,
                                            value=50,
                                            step=1)
                    starting_step = gr.Slider(
                        label="Step of MasaCtrl",
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
                    image_resolution = gr.Slider(label="Image Resolution",
                                                minimum=256,
                                                maximum=768,
                                                value=512,
                                                step=64)
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
            image_source = gr.Image(label="Source Image")
            image_fixed = gr.Image(label="Image with Fixed Seed")
            image_masactrl = gr.Image(label="Image with MasaCtrl")

        inputs = [
            source_prompt, target_prompt, starting_step, starting_layer,
            image_resolution, ddim_steps, scale, seed, appended_prompt,
            negative_prompt
        ]
        run_btn.click(consistent_synthesis, inputs,
                        [image_source, image_fixed, image_masactrl])

        gr.Examples(
            [[
                "1boy, bishounen, casual, indoors, sitting, coffee shop, bokeh",
                "1boy, bishounen, casual, indoors, standing, coffee shop, bokeh",
                42
            ],
                [
                    "1boy, casual, outdoors, sitting",
                    "1boy, casual, outdoors, sitting, side view", 42
                ],
                [
                    "1boy, casual, outdoors, sitting",
                    "1boy, casual, outdoors, standing, clapping hands", 42
                ],
                [
                    "1boy, casual, outdoors, sitting",
                    "1boy, casual, outdoors, sitting, shows thumbs up", 42
                ],
                [
                    "1boy, casual, outdoors, sitting",
                    "1boy, casual, outdoors, sitting, with crossed arms", 42
                ],
                [
                    "1boy, casual, outdoors, sitting",
                    "1boy, casual, outdoors, sitting, rasing hands", 42
                ]],
            [source_prompt, target_prompt, seed],
        )
    return demo


if __name__ == "__main__":
    demo_syntehsis = create_demo_synthesis()
    demo_synthesis.launch()
