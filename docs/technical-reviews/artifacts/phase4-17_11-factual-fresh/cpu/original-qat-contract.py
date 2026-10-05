class _QATLinear(nn.Linear):
    """與 QuantizedLinear 同一個 per-channel 規則，僅 backward 採 STE。"""

    def __init__(self, source, bits=4):
        super().__init__(source.in_features, source.out_features, source.bias is not None)
        self.weight = source.weight
        self.bias = source.bias
        self.bits = bits

    def forward(self, x):
        integers, scale = quantize_symmetric(self.weight.detach(), self.bits, per_channel=True)
        restored = integers * scale
        simulated = self.weight + (restored - self.weight).detach()
        return F.linear(x, simulated, self.bias)

def _qat_layers(model, remove=False):
    for name, child in list(model.named_children()):
        if remove and isinstance(child, _QATLinear):
            plain = nn.Linear(child.in_features, child.out_features, child.bias is not None)
            plain.weight, plain.bias = child.weight, child.bias
            setattr(model, name, plain)
        elif not remove and isinstance(child, nn.Linear):
            setattr(model, name, _QATLinear(child))
        else:
            _qat_layers(child, remove)
    return model

