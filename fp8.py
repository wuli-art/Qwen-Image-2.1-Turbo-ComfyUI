import torch
from torch import nn

from .quantized import load_quantized_transformer


class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device=device))

    def forward(self, hidden_states):
        if hidden_states.numel() == 0:
            return hidden_states.new_empty(*hidden_states.shape[:-1], self.out_features)
        flattened = hidden_states.reshape(-1, self.in_features).float()
        scale = (flattened.abs().amax(dim=-1, keepdim=True) / 448).clamp_min(1e-12)
        quantized = (flattened / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        output = torch._scaled_mm(
            quantized.contiguous(), self.weight.t(), scale_a=scale.contiguous(),
            scale_b=self.weight_scale.reshape(1, -1).contiguous(),
            out_dtype=hidden_states.dtype, use_fast_accum=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.reshape(*hidden_states.shape[:-1], self.out_features)


def check_fp8_device(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This FP8 checkpoint requires a CUDA GPU and a compatible PyTorch CUDA build.")
    try:
        with torch.inference_mode():
            probe = FP8Linear(32, 32).to(device)
            probe.weight.fill_(1)
            probe.weight_scale.fill_(1)
            result = probe(torch.ones(16, 32, device=device, dtype=torch.bfloat16))
            torch.cuda.synchronize(device)
            if not torch.isfinite(result).all() or not torch.allclose(result.float(), torch.full_like(result.float(), 32)):
                raise RuntimeError("FP8 kernel returned unexpected values")
    except Exception as error:
        raise RuntimeError(
            "The GPU/PyTorch combination cannot execute this per-channel FP8 kernel. "
            "Use a compatible CUDA build and GPU; this node does not silently fall back to BF16."
        ) from error


def load_fp8_transformer(model_path):
    expected = {"quant_method": "fp8", "fmt": "e4m3", "weight_scheme": "per_channel",
                "activation_scheme": "dynamic_per_token"}
    return load_quantized_transformer(model_path, FP8Linear, expected, torch.float8_e4m3fn)
