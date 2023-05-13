## MasaCtrl: Tuning-free <span style="text-decoration: underline"><font color="Tomato">M</font></span>utu<span style="text-decoration: underline"><font color="Tomato">a</font></span>l <span style="text-decoration: underline"><font color="Tomato">S</font></span>elf-<span style="text-decoration: underline"><font color="Tomato">A</font></span>ttention <span style="text-decoration: underline"><font color="Tomato">Control</font></span> for Consistent Image Synthesis and Editing

Pytorch implementation of [MasaCtrl: Tuning-free Mutual Self-Attention Control for **Consistent Image Synthesis and Editing**](https://arxiv.org/abs/2304.08465)

[Mingdeng Cao](https://github.com/ljzycmd),
[Xintao Wang](https://xinntao.github.io/),
[Zhongang Qi](https://scholar.google.com/citations?user=zJvrrusAAAAJ),
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ),
[Xiaohu Qie](https://scholar.google.com/citations?user=mk-F69UAAAAJ),
[Yinqiang Zheng](https://scholar.google.com/citations?user=JD-5DKcAAAAJ)

[![arXiv](https://img.shields.io/badge/ArXiv-2304.08465-brightgreen)](https://arxiv.org/abs/2304.08465)
[![Project page](https://img.shields.io/badge/Project-Page-brightgreen)](https://ljzycmd.github.io/projects/MasaCtrl/)
[![demo](https://img.shields.io/badge/Demo-Hugging%20Face-brightgreen)](https://huggingface.co/spaces/TencentARC/MasaCtrl)
[![demo](https://img.shields.io/badge/Demo-Colab-brightgreen)](https://colab.research.google.com/drive/1DZeQn2WvRBsNg4feS1bJrwWnIzw1zLJq?usp=sharing)

---

<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/overview.png">
<i> MasaCtrl enables performing various consistent non-rigid image synthesis and editing without fine-tuning and optimization. </i>
</div>


## Updates

- [2023/5/13] The inference code of MasaCtrl with T2I-Adapter is available.
- [2023/4/28] [Hugging Face demo](https://huggingface.co/spaces/TencentARC/MasaCtrl) released.
- [2023/4/25] Code released.
- [2023/4/17] Paper is available [here](https://arxiv.org/abs/2304.08465).


## Introduction

We propose MasaCtrl, a tuning-free method for non-rigid consistent image synthesis and editing. The key idea is to combine the `contents` from the *source image* and the `layout` synthesized from *text prompt and additional controls* into the desired synthesized or edited image, with **Mutual Self-Attention Control**.


## Main Features

### 1 Consistent Image Synthesis and Editing

MasaCtrl can perform prompt-based image synthesis and editing that changes the layout while maintaining contents of source image.

>*The target layout is synthesized directly from the target prompt.*

<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/results_synthetic.png">
<i>Consistent synthesis results</i>

<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/results_real.png">
<i>Real image editing results</i>
</div>


### 2 Integration to Controllable Diffusion Models

Directly modifying the text prompts often cannot generate target layout of desired image, thus we further integrate our method into existing proposed controllable diffusion pipelines (like T2I-Adapter and ControlNet) to obtain stable synthesis and editing results.

>*The target layout controlled by additional guidance.*

<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/results_w_adapter.png">
<i>Synthesis (left part) and editing (right part) results with T2I-Adapter</i>
</div>


### 3 Generalization to Other Models: Anything-V4

Our method also generalize well to other Stable-Diffusion-based models.

<div align="center">
<img src="https://huggingface.co/TencentARC/MasaCtrl/resolve/main/assets/anythingv4_synthetic.png">
<i>Results on Anything-V4</i>
</div>


## Usage

### Requirements
We implement our method with [diffusers](https://github.com/huggingface/diffusers) code base with similar code structure to [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt). The code runs on Python 3.8.5 with Pytorch 1.11. Conda environment is highly recommended.

```base
pip install -r requirements.txt
```

### Checkpoints

**Stable Diffusion:**
We mainly conduct expriemnts on Stable Diffusion v1-4, while our method can generalize to other versions (like v1-5).

You can download these checkpoints on their official repository and [Hugging Face](https://huggingface.co/).

**Personalized Models:**
You can download personlized models from [CIVITAI](https://civitai.com/) or train your own customized models.


### Demos

**Notebook demos**

To run the synthesis with MasaCtrl, single GPU with at least 16 GB VRAM is required. 

The notebook `playground.ipynb` and `playground_real.ipynb` provide the synthesis and real editing samples, respectively.

**Online demos**

We provide the [![demo](https://img.shields.io/badge/Demo-Hugging%20Face-brightgreen)](https://huggingface.co/spaces/TencentARC/MasaCtrl) demo with Gradio app. Note that you may copy the demo into your own space to use the GPU.

The [![demo](https://img.shields.io/badge/Demo-Colab-brightgreen)](https://colab.research.google.com/drive/1DZeQn2WvRBsNg4feS1bJrwWnIzw1zLJq?usp=sharing) is also available. 

**Local Gradio demo**

You can launch the provided Gradio demo locally with

```bash
CUDA_VISIBLE_DEVICES=0 python app.py
```


### MasaCtrl with T2I-Adapter

Install [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) and prepare the checkpoints following their provided tutorial. Assuming it has been successfully installed and the root directory is `T2I-Adapter`. 

Thereafter copy the core `masactrl` package and the inference code `masactrl_w_adapter.py` to the root directory of T2I-Adapter

```bash
cp MasaCtrl/masactrl T2I-Adapter/
cp masactrl_w_adapter.py T2I-Adapter/
```

Last, you can inference the images with following command (with sketch adapter)

```bash
python masactrl_w_adapter.py \
--which_cond sketch \
--cond_path_src SOURCE_CONDITION_PATH \
--cond_path CONDITION_PATH \
--cond_inp_type sketch \
--prompt_src "A bear walking in the forest" \
--prompt "A bear standing in the forest" \
--sd_ckpt models/sd-v1-4.ckpt \
--resize_short_edge 512 \
--cond_tau 1.0 \
--cond_weight 1.0 \
--n_samples 1 \
--adapter_ckpt models/t2iadapter_sketch_sd14v1.pth
```

## Acknowledgements

We thank awesome research works [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt), [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter).

## Citation

```bibtex
@misc{cao2023masactrl,
      title={MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing}, 
      author={Mingdeng Cao and Xintao Wang and Zhongang Qi and Ying Shan and Xiaohu Qie and Yinqiang Zheng},
      year={2023},
      eprint={2304.08465},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
```


## Contact

If your have any comments or questions, please [open a new issue](https://github.com/TencentARC/MasaCtrl/issues/new/choose) or feel free to contact [Mingdeng Cao](https://github.com/ljzycmd) and [Xintao Wang](https://xinntao.github.io/).