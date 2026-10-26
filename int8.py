import torch
from torch import nn
from torch.nn import functional

from .quantized import load_quantized_transformer


class INT8Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device=device))

    def forward(self, hidden_states):
        if hidden_states.numel() == 0:
            return hidden_states.new_empty(*hidden_states.shape[:-1], self.out_features)
        if hidden_states.device.type != "cuda" or hidden_states.dtype != torch.bfloat16:
            raise ValueError("INT8 inference requires CUDA BF16 activations")
        if self.weight.dtype != torch.int8 or self.weight_scale.dtype != torch.float32:
            raise ValueError("Move the model by device without casting its quantization buffers")
        activations = hidden_states.float().reshape(-1, self.in_features)
        activation_scale = (activations.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(1e-12)
        quantized = (activations / activation_scale).clamp(-127, 127).round().to(torch.int8)
        rows = quantized.shape[0]
        padding = max(32, ((rows + 15) // 16) * 16) - rows
        aligned = functional.pad(quantized, (0, 0, 0, padding)) if padding else quantized
        accumulators = torch._int_mm(aligned.contiguous(), self.weight.t())[:rows]
        output = (accumulators.float() * activation_scale * self.weight_scale.reshape(1, -1)).to(hidden_states.dtype)
        return output.reshape(*hidden_states.shape[:-1], self.out_features)


def check_int8_device(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This INT8 checkpoint requires a CUDA GPU and a compatible PyTorch CUDA build.")
    try:
        with torch.inference_mode():
            probe = INT8Linear(32, 32).to(device)
            probe.weight.fill_(1)
            probe.weight_scale.fill_(1)
            result = probe(torch.ones(1, 32, device=device, dtype=torch.bfloat16))
            torch.cuda.synchronize(device)
            if not torch.isfinite(result).all() or not torch.allclose(result.float(), torch.full_like(result.float(), 32)):
                raise RuntimeError("INT8 kernel returned unexpected values")
    except Exception as error:
        raise RuntimeError(
            "The GPU/PyTorch combination cannot execute this W8A8 torch._int_mm kernel. "
            "Use a compatible CUDA build and GPU; this node does not silently fall back to BF16."
        ) from error


def load_int8_transformer(model_path):
    expected = {"quant_method": "wuli_int8", "weight_dtype": "int8", "activation_dtype": "int8",
                "weight_scheme": "per_channel", "activation_scheme": "dynamic_per_token",
                "weight_scale_dtype": "float32"}
    return load_quantized_transformer(model_path, INT8Linear, expected, torch.int8, initial_dtype=torch.bfloat16)
