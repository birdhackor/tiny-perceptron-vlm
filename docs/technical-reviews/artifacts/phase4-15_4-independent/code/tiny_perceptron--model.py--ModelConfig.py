class ModelConfig:
    vocab_size: int = 264
    width: int = 32
    layers: int = 1
    heads: int = 1
    max_length: int = 128
    norm: str = "layer"
    activation: str = "gelu"
    rotary: bool = False
    tied: bool = False
    kv_heads: int | None = None
    backend: str = "manual"
    experts: int = 0
    top_k: int = 2
