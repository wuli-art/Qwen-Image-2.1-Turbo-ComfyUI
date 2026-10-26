from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


MODEL_REPO = "Wuli-art/Qwen-Image-2.1-Turbo"
BASE_REPO = "Qwen/Qwen-Image-2.1"
VARIANTS = tuple(f"{steps}step-{precision}" for precision in ("fp8", "int8", "bf16") for steps in (8, 4, 2))
BASE_PATTERNS = ["model_index.json", "processor/*", "text_encoder/*", "vae/*"]


def download_snapshot(repo_id, patterns):
    print(f"[Qwen Image 2.1 Turbo] Preparing {repo_id}: checking cache / downloading missing files.", flush=True)
    try:
        return Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))
    except LocalEntryNotFoundError as error:
        raise RuntimeError(
            f"Cannot download {repo_id}, and the required snapshot is not cached. "
            "Check your network or set local model paths. For offline use, download the files first."
        ) from error
    except HfHubHTTPError as error:
        raise RuntimeError(
            f"Cannot download {repo_id}. Check network access and repository access. "
            "For private or gated models, run `hf auth login` in the ComfyUI environment "
            "or set HF_TOKEN before starting ComfyUI."
        ) from error


def resolve_paths(model_path, base_model_path, models_dir, variant, check_interrupt):
    def local_path(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path(models_dir) / path

    check_interrupt()
    base = local_path(base_model_path.strip()) if base_model_path.strip() else None
    if base is not None and not base.is_dir():
        raise ValueError(f"Local base_model_path does not exist: {base}")
    if model_path.strip():
        model = local_path(model_path.strip())
        if not model.is_dir():
            raise ValueError(f"Local model_path does not exist: {model}. Clear it to download automatically.")
    else:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown model variant: {variant!r}")
        model = download_snapshot(MODEL_REPO, [f"{variant}/*.json", f"{variant}/*.safetensors"]) / variant
    check_interrupt()
    if base is None:
        base = model if (model / "model_index.json").is_file() else download_snapshot(BASE_REPO, BASE_PATTERNS)
    check_interrupt()
    return model, str(base)
