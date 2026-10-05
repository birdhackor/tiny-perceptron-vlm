# 18.2 independent parameter derivation

For the explicit untied, learned-position, LayerNorm, GELU DenseFFN TinyLM defaults with vocabulary V=264 and maximum positions M=128, width w and layer count L:

- Token embedding: 264w. Position embedding: 128w. Untied output matrix: 264w. Final LayerNorm weight/bias: 2w. These give 658w and do not repeat with L.
- In each layer, full MHA has Q,K,V,O matrices: 4w². DenseFFN expands to 4w: up/down matrices total 8w², biases total 5w. Two LayerNorms have weights/biases totaling 4w. Each layer totals 12w²+9w.
- Total N(w,L)=658w+L(12w²+9w). Heads=2 with kv_heads=None changes head partitioning, not these full projection sizes.

N(16,2)=16960, N(8,1)=6104, N(16,1)=13744, N(64,2)=141568, N(32,1)=33632. The 8-wide student/16-wide two-layer teacher ratio is 6104/16960≈0.35990566, not 1/8. At width16, one-layer/two-layer ratio is 13744/16960≈0.81037736, not 1/2.

A byte has 8 bits. FP32 means a 32-bit scalar, hence 4 bytes. For all inspected model Parameters, torch.float32 and element_size()==4 were asserted, so numeric parameter bytes equal N×4 exactly: 67840,24416,54976,566272,134528 respectively. This excludes file headers, buffers, optimizer states, cache, activations and peak runtime memory.

The original Transformer paper ties embedding/output weights; the counted repository TinyLM has tied=False, so its embedding and output matrices are separately counted. The formula is specific to these defaults, not a universal Transformer formula.

This proof includes a forward pass only to validate axes [batch=1,positions=3,width=w] and [batch=1,positions=3,vocabulary=264]; it computes no accuracy or deployment speed and makes no quality claim.
