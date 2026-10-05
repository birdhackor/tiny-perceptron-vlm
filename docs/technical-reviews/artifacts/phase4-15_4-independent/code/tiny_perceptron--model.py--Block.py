class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm = nn.LayerNorm if config.norm == "layer" else RMSNorm
        self.norm1, self.norm2 = norm(config.width), norm(config.width)
        self.attention = CausalAttention(config.width, config.heads, config.kv_heads, config.backend, config.rotary)
        self.ffn = (
            MoEFFN(config.width, config.experts, config.top_k)
            if config.experts
            else DenseFFN(config.width, activation=config.activation)
        )

    def forward(self, x, valid=None, positions=None, segments=None, cache=None):
        attended, new_cache = self.attention(self.norm1(x), valid, positions, segments, cache)
        x = x + attended
        h = self.ffn(self.norm2(x))
        auxiliary = x.new_zeros(())
        if isinstance(h, tuple):
            h, auxiliary, _ = h
        return x + h, new_cache, auxiliary
