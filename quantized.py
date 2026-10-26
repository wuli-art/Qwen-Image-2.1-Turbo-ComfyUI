import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torch import nn


def transformer_folder(model_path):
    root = Path(model_path)
    return root / "transformer" if (root / "transformer").is_dir() else root


def load_quantized_transformer(model_path, linear_class, expected_quantization, weight_dtype, initial_dtype=None):
    from diffusers import QwenImage21Transformer2DModel

    folder = transformer_folder(model_path)
    config = json.loads((folder / "config.json").read_text())
    quantization = config.pop("quantization_config", {})
    if any(quantization.get(key) != value for key, value in expected_quantization.items()):
        raise ValueError(f"Unsupported quantization layout; expected {expected_quantization}")
    if config.pop("affine_layer_norm", False) is not True:
        raise ValueError("This loader requires the exported affine_layer_norm=true checkpoint.")
    with init_empty_weights(include_buffers=False):
        model = QwenImage21Transformer2DModel.from_config(config)
        if initial_dtype is not None:
            model.to(dtype=initial_dtype)
        for block in model.transformer_blocks:
            for name in ("img_norm1", "img_norm2"):
                original = getattr(block, name)
                setattr(block, name, nn.LayerNorm(
                    original.normalized_shape, eps=original.eps, elementwise_affine=True,
                    bias=False, dtype=torch.bfloat16,
                ))
            for name in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                         "img_mlp.proj", "img_mlp.out", "img_mlp.gate_layer"):
                parent_name, attribute = name.rsplit(".", 1)
                parent = block.get_submodule(parent_name)
                original = getattr(parent, attribute)
                if original.bias is not None:
                    raise ValueError(f"Unexpected bias in {name}")
                setattr(parent, attribute, linear_class(original.in_features, original.out_features, device="meta"))
    mapping = json.loads((folder / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    expected_keys = set(model.state_dict())
    if set(mapping) != expected_keys:
        raise ValueError(f"Checkpoint keys differ: missing={sorted(expected_keys - set(mapping))}, "
                         f"unexpected={sorted(set(mapping) - expected_keys)}")
    for filename in sorted(set(mapping.values())):
        relative = Path(filename)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in filename or ":" in filename
                or relative.suffix != ".safetensors"):
            raise ValueError("Checkpoint shard must use a relative safetensors path without parent traversal.")
        shard_path = folder / relative
        shard = load_file(str(shard_path))
        if set(shard) != {key for key, value in mapping.items() if value == filename}:
            raise ValueError(f"Shard keys do not match index: {filename}")
        model.load_state_dict(shard, strict=False, assign=True)
        del shard
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError("Uninitialized model tensors")
    for module in model.modules():
        if isinstance(module, linear_class):
            if module.weight.dtype != weight_dtype or module.weight_scale.dtype != torch.float32:
                raise ValueError("Quantized weight or scale dtype does not match the checkpoint layout")
            if not torch.isfinite(module.weight_scale).all() or not (module.weight_scale > 0).all():
                raise ValueError("Quantization scales must be finite and positive")
    return model.eval().requires_grad_(False)
