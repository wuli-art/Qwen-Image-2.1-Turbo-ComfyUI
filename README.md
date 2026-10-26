# Qwen Image 2.1 Turbo for ComfyUI

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Wuli-art/Qwen-Image-2.1-Turbo)

A ComfyUI node for `Wuli-art/Qwen-Image-2.1-Turbo`, supporting text-to-image generation and reference-image editing.

- BF16, FP8 and INT8 with 2-, 4- and 8-step variants.
- Automatic downloads on first use, with Hugging Face cache reuse.
- CPU offload and optional local model paths.

## Installation

Requires Python 3.10+, an NVIDIA CUDA GPU and a compatible PyTorch build. Your GPU must support the selected precision's operations.

Place this folder in `ComfyUI/custom_nodes/`, then install dependencies using the Python environment that runs ComfyUI:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-Qwen-Image-2.1-Turbo/requirements.txt
```

For Windows portable, run from the portable installation's root:

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-Qwen-Image-2.1-Turbo\requirements.txt
```

Restart ComfyUI after installation.

## Usage

1. Import `example_workflows/text_to_image.json`. For editing, import `example_workflows/image_reference.json` and select a reference image.
2. Leave `model_path` and `base_model_path` empty, then choose a model under `variant`.
3. Enter a prompt and run the workflow.

The node downloads the selected Turbo weights and the text encoder, processor and VAE from `Qwen/Qwen-Image-2.1`. Base components are shared across variants. Initial downloads require time and disk space; progress appears in the ComfyUI terminal.

Use the default `cpu_offload` mode to reduce VRAM usage, or select `gpu` if you have enough VRAM. Enable `vae_tiling` to reduce decoding memory at higher resolutions; it may change the output.

## Optional settings

- **Local models:** Set `model_path` to override `variant`. Use `base_model_path` for local base components. Paths can be absolute or relative to `ComfyUI/models/`.
- **Cache:** Set `HF_HOME` or `HF_HUB_CACHE` before starting ComfyUI. After downloading all required files, set `HF_HUB_OFFLINE=1` for offline use.
- **Repository access:** If authentication is required, run `hf auth login` in the same environment or set `HF_TOKEN`. Do not store tokens in workflows.
- **Stalled downloads:** If Xet transfers are unavailable, set `HF_HUB_DISABLE_XET=1` before startup.

This node runs a complete Diffusers pipeline. Native ComfyUI MODEL/CLIP/VAE outputs and LoRA integration are not supported. Model weights remain subject to the terms of their respective repositories.
