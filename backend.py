import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .fp8 import check_fp8_device, load_fp8_transformer
from .int8 import check_int8_device, load_int8_transformer
from .quantized import transformer_folder


def read_model_config(model_path):
    root = Path(model_path).expanduser().resolve()
    complete = (root / "model_index.json").is_file()
    required = ("transformer/config.json", "processor", "text_encoder", "vae", "scheduler") if complete else (
        "config.json", "sampling_config.json", "scheduler_config.json",
    )
    for relative in required:
        if not (root / relative).exists():
            raise ValueError(f"Missing {relative} in {root}. Select a complete export or a variant folder such as 8step-int8.")
    config = json.loads((root / ("model_index.json" if complete else "sampling_config.json")).read_text())
    if complete and config.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("Expected a QwenImage21Pipeline export.")
    sigmas = config.get("sample_sigmas")
    if not isinstance(sigmas, list) or len(sigmas) not in (2, 4, 8):
        raise ValueError("This node requires an exported 2-, 4-, or 8-step sample_sigmas schedule.")
    if not complete and (config.get("num_inference_steps") != len(sigmas) or config.get("true_cfg_scale") != 1.0):
        raise ValueError("Sampling config must match its sigma count and use true_cfg_scale=1.0.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or not 0 < value <= 1 for value in sigmas):
        raise ValueError("sample_sigmas must be finite values in (0, 1].")
    if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError("sample_sigmas must be strictly descending.")
    return root, sigmas


def model_precision(model_path):
    config = json.loads((transformer_folder(model_path) / "config.json").read_text())
    quantization = config.get("quantization_config")
    if quantization is None and config.get("_class_name") == "QwenImage21Transformer2DModel":
        return "bf16"
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    if method not in ("fp8", "wuli_int8"):
        raise ValueError(f"Unsupported quant_method {method!r}; select a BF16, FP8 or Wuli INT8 export.")
    return "int8" if method == "wuli_int8" else "fp8"


def load_bf16_transformer(model_path):
    from diffusers import QwenImage21Transformer2DModel

    return QwenImage21Transformer2DModel.from_pretrained(
        str(transformer_folder(model_path)), torch_dtype=torch.bfloat16, local_files_only=True,
    ).eval().requires_grad_(False)


def base_components(model_path, base_model_path=""):
    root = Path(base_model_path).expanduser().resolve() if base_model_path else Path(model_path)
    for relative in ("model_index.json", "processor", "text_encoder", "vae"):
        if not (root / relative).exists():
            raise ValueError(
                f"Missing base component {relative} in {root}. For a downloaded variant directory, "
                "set base_model_path to a local Qwen-Image-2.1 model or a complete existing export."
            )
    config = json.loads((root / "model_index.json").read_text())
    if config.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("Base components must belong to QwenImage21Pipeline.")
    return root


def reference_images(images):
    if images is None:
        return None
    if images.ndim != 4 or images.shape[-1] not in (3, 4) or len(images) == 0:
        raise ValueError("Reference IMAGE must be a non-empty BHWC RGB/RGBA tensor.")
    if not torch.isfinite(images).all():
        raise ValueError("Reference images contain non-finite pixels.")
    pixels = (images.detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return [Image.fromarray(image) for image in pixels]


def load_pipeline(model_path, device, memory_mode="cpu_offload", vae_tiling=False, base_model_path=""):
    root, sigmas = read_model_config(model_path)
    if memory_mode not in ("cpu_offload", "gpu"):
        raise ValueError("memory_mode must be cpu_offload or gpu")
    base = base_components(root, base_model_path)
    precision = model_precision(root)
    if precision == "int8":
        check_int8_device(device)
    elif precision == "fp8":
        check_fp8_device(device)
    else:
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("BF16 inference requires a CUDA device.")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("The selected CUDA device does not support native BF16 inference.")
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline

    loaders = {"bf16": load_bf16_transformer, "fp8": load_fp8_transformer, "int8": load_int8_transformer}
    transformer = loaders[precision](root)
    scheduler_folder = root / "scheduler" if (root / "model_index.json").is_file() else root
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_folder), local_files_only=True)
    pipeline = QwenImage21Pipeline.from_pretrained(
        str(base), transformer=transformer, scheduler=scheduler, dtype=torch.bfloat16, local_files_only=True,
    )
    pipeline._qwen_turbo_precision = (
        "INT8 W8A8 per-channel weights / dynamic per-token activations" if precision == "int8" else
        "FP8 E4M3 per-channel weights / dynamic per-token activations" if precision == "fp8" else
        "BF16 unquantized weights / activations"
    )
    if vae_tiling:
        pipeline.vae.enable_tiling()
    pipeline.set_progress_bar_config(disable=True)
    if memory_mode == "cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)
    return pipeline, sigmas


def generate(pipeline, sigmas, prompt, width, height, seed, device, reference=None, callback=None):
    if width % 16 or height % 16 or min(width, height) < 256:
        raise ValueError("Width and height must be multiples of 16 and at least 256.")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipeline(
            prompt=prompt, image=reference_images(reference), width=width, height=height,
            num_inference_steps=len(sigmas), sigmas=list(sigmas), true_cfg_scale=1.0,
            generator=torch.Generator(device=device).manual_seed(seed), output_type="pil",
            callback_on_step_end=callback,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    images = torch.from_numpy(np.stack([
        np.asarray(image.convert("RGBA"), dtype=np.float32) / 255 for image in result.images
    ]))
    metrics = {
        "inference_seconds": round(elapsed, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3),
        "steps": len(sigmas), "seed": seed, "width": width, "height": height,
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "precision": pipeline._qwen_turbo_precision,
    }
    return images, metrics


def release_pipeline(pipeline, move_to_cpu=True):
    if pipeline is not None:
        pipeline.remove_all_hooks()
        if move_to_cpu:
            pipeline.to(device="cpu")
    gc.collect()
