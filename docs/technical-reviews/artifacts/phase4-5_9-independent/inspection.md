# 5.9 fresh independent inspection

Reviewer task: `/root/phase4_factual_coordinator/factual_5_9`; date: 2026-10-05.
No previous technical or reader report was read. This section is not the chapter's first section.

Read the complete current 5.9, the complete T.4 recipe as contextual documentation,
the review method, schema, extraction/execution helper, and review protocol.
Read TinyLM/ModelConfig/description/forward, CausalAttention/manual_attention,
DenseFFN, ByteTokenizer/shifted/pad_batch, and the historical text_foundation,
fit_lm, and experiment runner timing contracts. An incidental current chapter
line-range read also displayed lines 470–555; that content is not used as evidence.

No SVG, raster, or diagram is referenced in 5.9. Thus no figure render/view was
applicable. The section's sequence lengths and timing boundaries are stated in
text and executable code; no visual result is claimed.

## Authoritative sources personally inspected

All PyTorch files were retrieved from the upstream pytorch/pytorch repository at
5c4886908584029761b579af026dcfb627c84070, exactly the git revision reported by
the installed 2.14.1+cpu wheel. These are upstream original sources, not search
snippets. URLs/hashes/date and failed retrieval are in the fetch receipts.

- torch-tensor-docs.py lines 1728–1742: Tensor.element_size reports bytes;
  lines 6555–6563: .grad is populated/accumulated by backward;
  lines 6753–6768: itemsize and nbytes = numel * element_size.
- torch-adam.py lines 148–192: parameters with gradients have state initialized;
  exp_avg and exp_avg_sq are tensors shaped like each parameter. No universal
  fixed total-training-memory multiplier is asserted.
- torch-autograd.md lines 12–54: forward records a reverse-mode graph and saves
  intermediate tensors when backward needs them. The original .rst URL was 404;
  the same pinned revision's .md source was then retrieved and read.
- torch-grad-mode.py lines 22–86: no_grad disables reverse-mode gradient tracking
  and restores prior state on exit; factory and forward-mode exceptions are
  stated. The section's ordinary TinyLM forward falls within this API scope.
- torch-module.py lines 2894–2932: train changes the training flag recursively;
  eval is train(False), with effects on selected modules. TinyLM has no Dropout
  or BatchNorm here; eval alone still leaves its output differentiable, as checked.
- torch-benchmark-timer.py lines 67–136: warmup, controlled thread count,
  synchronization of asynchronous accelerators, and repeated measurements
  address lazy initialization and noise.
- torch-cuda.py lines 1271–1281: synchronize waits for kernels in all streams on
  the chosen CUDA device. This is source verification only; no CUDA was run.
- python-time.rst (official CPython v3.13.5) lines 321–347: perf_counter returns
  fractional seconds, uses a high-resolution monotonic clock, and supports
  elapsed intervals via differences between calls.

## Counts and scopes

width=8 with default vocab=264, positions=128, layers=heads=1, untied FP32:
embedding 264*8=2112; position 128*8=1024; Q/K/V/out 4*8*8=256;
FFN (8*32+32)+(32*8+8)=552; three affine LayerNorms 3*2*8=48;
output 8*264=2112. Total 6104; 6104*4=24416 bytes. These are tensor
parameter data bytes, not the serialized checkpoint size or total resident memory.

The unchanged original fence executed once with one warmup and 10 measured
forwards; the only-change six-position fence also executed with the same policy.
The additional bounded CPU check performed five blocks of 100 forwards for each
length with one thread, to inspect variation; it is not an estimate of hardware
speedup. Observed real attention weights were [1,1,3,3] and [1,1,6,6].
No backward or optimizer update occurred. Both lengths retain 6104 FP32 parameters.

Original result is docs/course-experiments/results/text_foundation.json,
revision b52935d99f58b694cd932ac158c10c4d8d92d4c2, NVIDIA L4,
torch=2.14.1+cu126, Python=3.13.3, seed=42. Historical git files were recovered
and each file covered by the result's code_sha256 matched that recorded hash.
The generator is not in that recorded code hash map; its original revision is
nonetheless fixed by git, and reconstructed train/validation/test JSONL hashes
all match the result. No weights were loaded, copied, downloaded, or trained.

The historical default model width64/layers2/vocab264/max_length128 reconstructs
141568 FP32 parameters and 566272 raw parameter bytes. The historical sampling
algorithm was replayed only for counts using seed42, 600 draws of 16 examples
with replacement. All 9 documents fit in one chunk each; label lengths are
37,37,35,34,35,36,35,36,36. There are 9600 document exposures and 342462
effective targets including repeated bytes and EOS; these are not unique texts
or documents. Padding labels use IGNORE=-100 and are excluded. No training was
rerun to obtain the historical 6.1885463640000005 or 16.035376792999998 seconds.

Historical common.py lines 125–168 times the synchronized main 600-update
loop including batch preparation, logging, and periodic checkpoints. Initial
NLL precedes it; final training NLL and final model checkpoint follow it.
Historical run.py lines 71–107 times the experiment function through the final
CUDA synchronization, including evaluation, local saves, four scaling runs,
causality and resume probes. It excludes code hashing/import setup, final
result serialization, image build/startup, and external backup/upload. The
section says it also includes the four comparisons/evaluation/saving; it does
not claim those are the exhaustive contents. Four-decimal rounding agrees:
6.1885 and 16.0354 seconds. The section explicitly prevents cross-scope and
CPU-forward-versus-GPU-training speed comparisons.

Conclusion for this frozen section: pass; no substantive unresolved claim was
found. Timing is machine/load-specific and one warmup plus ten calls remains
a process demonstration, as the section itself says.
