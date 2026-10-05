# 18.14 independent storage derivation

All storage units below are bytes, not decimal kilobytes, MiB, serialized file size, or runtime peak memory. The numerator and denominator count every parameter and registered buffer in these untied TinyLM structures. Weights, biases, LayerNorm coefficients and scales start in FP32 (4 bytes per element). Vocabulary V=264, learned positional table P=128, width d, L blocks, dense FFN hidden width 4d.

Floating elements: token embedding 264d + position embedding 128d + output projection 264d + final LayerNorm 2d + L × (two LayerNorms 4d + four attention matrices 4d² + FFN matrices 8d² + FFN biases 5d). Thus the total is 658d + L(12d²+9d), each times 4 bytes.

- Teacher d=16,L=2: (10,528 + 6,432) × 4 = 67,840 bytes.
- Student d=8,L=1: (5,264 + 840) × 4 = 24,416 bytes.
- Quantized student retains embeddings and normalization: (394+4L)d × 4 = 12,736 bytes. FFN biases: 5Ld × 4 = 160 bytes. Per-output-channel scale tables: (9Ld+264) × 4 = 1,344 bytes. Linear weights have 12Ld²+264d = 2,880 elements, requiring 1,440 packed4 bytes or 2,880 int8 bytes. Sum: 15,680 or 17,120 bytes.
- Ratio uses the quantized student as numerator and teacher as denominator: 15,680 / 67,840 = 0.23113207547169812; rounding to four decimals gives 0.2311 (rounding error below 0.00005). This is greater than 24,416/8=3,052 bytes; an eightfold whole-model reduction is not implied.
- Existing run teacher d=64,L=2: 141,568 FP32 elements × 4 = 566,272 bytes.
- Existing run student d=32,L=1: 33,632 FP32 elements × 4 = 134,528 bytes.
- Its packed4 copy retains 50,944 FP32 parameter bytes; 640 bias bytes + 2,208 scale bytes + 10,368 packed weight bytes = 13,216 buffer bytes. Total: 64,160 bytes.

All relevant linear weight dimensions are even, so this example has no odd-element nibble padding overhead. The CPU check enumerates named parameters and buffers independently, checks exact integer equality against these formulas, and checks output activations remain FP32 in a short random-structure forward pass. It does not score trained models.

MAE example: original [[1,-2],[3,-4]], restored [[0.5,-1.5],[3,-3]]; absolute errors [0.5,0.5,0,1], denominator 4 elements, sum 2, mean 0.5. This has units of weight values and does not represent answer exact-match fraction, distillation loss, file bytes, or time.

For the existing attribute test, all four versions use 10 records from the same two held-out families. Reaggregating saved generated IDs and reference IDs gives teacher 5/10, CE student 4/10, CE+KL student 4/10, packed4 CE+KL student 3/10. Each has 69 teacher-forced supervised tokens including EOS and a generation cap of 24 new tokens per record; every saved response contains EOS. These are provenance checks of an existing single-seed run, not newly generated predictions. The changed color question has exact byte IDs for `blue` before conversion and `ble` after conversion.
