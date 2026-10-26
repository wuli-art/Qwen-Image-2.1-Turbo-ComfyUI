import gc
import json
import time

from .backend import generate, load_pipeline, release_pipeline
from .download import VARIANTS, resolve_paths


class QwenImage21TurboGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "tooltip": "Leave empty to download the selected variant automatically. A local path overrides variant."}),
                "prompt": ("STRING", {"multiline": True, "default": "A ceramic teapot on a wooden table"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xffffffffffffffff}),
                "memory_mode": (["cpu_offload", "gpu"],),
                "vae_tiling": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference": ("IMAGE",),
                "base_model_path": ("STRING", {"default": "", "tooltip": "Optional local base components. Leave empty to download automatically or use a complete export."}),
                "variant": (list(VARIANTS), {"default": "8step-fp8", "tooltip": "Model to download when model_path is empty."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "performance")
    FUNCTION = "execute"
    CATEGORY = "Qwen Image 2.1 Turbo"
    DESCRIPTION = "Select a variant to download automatically on first use. Cached files are reused. RGBA output preserves transparency when connected to Save Image. Optional local paths override downloads; IMAGE batch supplies references."

    def execute(self, model_path, prompt, width, height, seed, memory_mode, vae_tiling, reference=None, base_model_path="", variant="8step-fp8"):
        import folder_paths
        import comfy.model_management as management
        from comfy.utils import ProgressBar

        started = time.perf_counter()
        path, base_model_path = resolve_paths(
            model_path, base_model_path, folder_paths.models_dir, variant,
            management.throw_exception_if_processing_interrupted,
        )
        download_seconds = time.perf_counter() - started
        device = management.get_torch_device()
        management.throw_exception_if_processing_interrupted()
        management.unload_all_models()
        management.soft_empty_cache()
        def callback(pipeline, step, timestep, callback_kwargs):
            management.throw_exception_if_processing_interrupted()
            progress.update_absolute(step + 1, len(sigmas))
            return callback_kwargs

        pipeline = None
        completed = False
        load_started = time.perf_counter()
        try:
            pipeline, sigmas = load_pipeline(path, device, memory_mode, vae_tiling, base_model_path)
            progress = ProgressBar(len(sigmas))
            load_seconds = time.perf_counter() - load_started
            management.throw_exception_if_processing_interrupted()
            images, metrics = generate(
                pipeline, sigmas, prompt, width, height, seed, device, reference, callback,
            )
            metrics.update(download_seconds=round(download_seconds, 3), load_seconds=round(load_seconds, 3),
                           memory_mode=memory_mode, vae_tiling=vae_tiling)
            completed = True
        finally:
            try:
                release_pipeline(pipeline, move_to_cpu=not completed)
            finally:
                pipeline = None
                gc.collect()
                management.soft_empty_cache()
        metrics["total_seconds_including_load_and_release"] = round(time.perf_counter() - started, 3)
        report = json.dumps(metrics, ensure_ascii=False, indent=2)
        print(f"[Qwen Image 2.1 Turbo] {report}")
        return images, report


NODE_CLASS_MAPPINGS = {"QwenImage21TurboGenerate": QwenImage21TurboGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenImage21TurboGenerate": "Qwen Image 2.1 Turbo · BF16 / FP8 / INT8"}
