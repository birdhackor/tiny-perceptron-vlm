
53: class TinyLM(nn.Module):
54:     def __init__(self, config=None):
55:         super().__init__()
56:         self.config = config or ModelConfig()
57:         c = self.config
58:         if c.norm not in ("layer", "rms") or c.layers < 1 or c.max_length < 1:
59:             raise ValueError("無效的 norm/layers/max_length 設定")
60:         self.embedding = nn.Embedding(c.vocab_size, c.width)
61:         self.position = None if c.rotary else nn.Embedding(c.max_length, c.width)
62:         self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
63:         self.final_norm = nn.LayerNorm(c.width) if c.norm == "layer" else RMSNorm(c.width)
64:         self.output = nn.Linear(c.width, c.vocab_size, bias=False)
65:         if c.tied:
66:             self.output.weight = self.embedding.weight
