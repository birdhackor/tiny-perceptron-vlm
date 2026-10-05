"""先教數值和真的 packing；參考運算會反量化，不宣稱低位元 kernel 加速。"""

import torch
from torch import nn
from torch.nn import functional as F


def quantize_symmetric(weight, bits=8, per_channel=False):
    if bits not in (4, 8):
        raise ValueError("教材支援 4 或 8 bit")
    maximum = 2 ** (bits - 1) - 1
    if per_channel:
        scale = weight.abs().amax(-1, keepdim=True).clamp(min=1e-8) / maximum
    else:
        scale = weight.abs().max().clamp(min=1e-8) / maximum
    integers = (weight / scale).round().clamp(-maximum, maximum).to(torch.int8)
    return integers, scale


def quantize_affine(x, bits=8):
    maximum = 2**bits - 1
    low, high = torch.minimum(x.min(), x.new_zeros(())), torch.maximum(x.max(), x.new_zeros(()))
    scale = (high - low).clamp(min=1e-8) / maximum
    zero = (-low / scale).round().clamp(0, maximum)
    q = (x / scale + zero).round().clamp(0, maximum).to(torch.uint8)
    return q, scale, zero


def pack_int4(integers):
    flat = integers.flatten().to(torch.int16)
    if ((flat < -8) | (flat > 7)).any():
        raise ValueError("int4 可儲存 -8 到 7")
    values = (flat + 8).to(torch.uint8)
    if values.numel() % 2:
        values = F.pad(values, (0, 1), value=8)
    return values[0::2] | (values[1::2] << 4)


def unpack_int4(packed, shape):
    values = torch.stack((packed & 15, packed >> 4), -1).flatten().to(torch.int16) - 8
    count = 1
    for dimension in shape:
        count *= dimension
    return values[:count].reshape(shape).to(torch.int8)


class QuantizedLinear(nn.Module):
    """真正壓縮儲存，forward 仍反量化成浮點作參考。"""

    def __init__(self, layer, bits=4):
        super().__init__()
        q, scale = quantize_symmetric(layer.weight.detach(), bits, per_channel=True)
        self.bits, self.shape = bits, list(q.shape)
        self.register_buffer("values", pack_int4(q) if bits == 4 else q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if layer.bias is None else layer.bias.detach().clone())

    def forward(self, x):
        q = unpack_int4(self.values, self.shape) if self.bits == 4 else self.values
        weight = (q * self.scale).to(x.dtype)
        return F.linear(x, weight, None if self.bias is None else self.bias.to(x.dtype))

    def storage_bytes(self):
        return sum(b.numel() * b.element_size() for b in self.buffers())


def replace_linear_layers(module, bits=4):
    """就地替換；tied embedding 的語意要在比較中明示，不自動維持共享。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinear(child, bits))
        else:
            replace_linear_layers(child, bits)
    return module


def fake_quantize(x, bits=8):
    q, scale = quantize_symmetric(x.detach(), bits)
    reconstructed = q * scale
    # straight-through estimator：forward 見量化誤差，backward 近似為恒等。
    return x + (reconstructed - x).detach()
