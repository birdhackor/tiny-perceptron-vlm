# 5.9 same-owner continuity callback

Reviewer: `/root/phase4_factual_coordinator/factual_5_9`, 2026-10-05.
The complete current 5.9 and current review method were personally read. The
previous canonical PASS was then copied byte-for-byte to a unique history file
before any substantive callback verification or report update; the exact order
is recorded in prior-pass-archive.json rather than retroactively changed.
No author repair explanation, third reader, or other reviewer answer was read.

The only 5.9 text difference is its second paragraph, defining byte/bit/FP32/dtype.
The entire current section, unchanged original Python fence, necessary T.4
context, source scripts, and official source snapshots were inspected/version
checked. T.4's width32 CPU recipe is distinct from the fixed text_foundation
width64/layers2 result; the current T.4 bytes exactly match my original input.
Current TinyLM/attention/DenseFFN/data files also exactly match my original
input hashes. No figure is referenced; no rendering or image view applies.

Personally re-read source code using AST-located classes/functions:
ModelConfig15–28, Block31–50, TinyLM53–89; manual_attention21–28 and
CausalAttention31–74; DenseFFN35–53. Official already-retrieved PyTorch sources
re-read: no_grad22–42, Module.train/eval2894–2932, autograd saved tensors39–54,
Adam initialization151–185, Tensor.element_size1728–1742 and .grad6555–6563,
benchmark Timer67–89, CUDA synchronize1271–1281. CPython perf_counter321–339
was re-read. There is no new gradient norm or gradient clipping explanation in
5.9, and no such mechanism is introduced by this callback. The gradient/storage
claim remains supported by backward-populated .grad, Adam's separate state, and
autograd's necessary saved intermediates. no_grad differs from eval; neither
the current example nor callback calls backward or an optimizer.

The new terminology was checked from primary files, not inferred from repair
notes. Official CPython v3.13.5 stdtypes.rst2721–2775 defines bytes as single
bytes with integer values 0<=x<256 and two hexadecimal digits per byte. Thus
256=2^8 alternatives give 8 bits per byte. The already-read official PyTorch
Tensor.element_size documents individual element size in bytes. The original
installed official PyTorch 2.14.1+cpu API declaration `_C/__init__.pyi`198–235
declares dtype, float32:dtype and finfo.bits. The original scalar type mapping
in bundled torch/headeronly/core/ScalarType.h64–77 maps Byte to uint8_t and
Float to float. Both original package files are permanently copied and verified
byte-identical, with package version/git revision and original paths recorded.
The short CPU probe confirms torch.float32 is a torch.dtype,
finfo(torch.float32).bits=32 and float32 Tensor.element_size()=4; 32/8=4.
This supports the new plain-language definition within the actual PyTorch/CPU
context. It does not assert every C/C++ implementation has the same float size.

Several attempted external dtype table URLs returned HTTP503 and were not read;
all attempts and stderr are retained. The successfully retrieved CPython primary
file and original official installed PyTorch API/source files plus measured
format sizes are sufficient for these narrow claims; no failed URL is recorded
as verified evidence or used to guess a table's contents.

The original current fence was actually executed again under a 15-second bound,
CPU/one thread and offline flags: 6104 parameters, 24416 raw FP32 bytes;
one warmup plus ten measured forwards, no backward or updates. It produced
0.0005230391005170531 seconds per forward in this run, a machine/load-specific
value, not a standardized performance answer. The unchanged original six-token
exercise and its earlier true attention-shape/variation evidence remain valid
because their inputs and implementation hashes are unchanged.

Original JSON was accessed by top-level key/type inventory followed only by
the named pointers in original-json-pointer-check.json. Its full bytes equal
my initial saved original input: no notes/review/HF backup commentary or
scope-correction summary was read during callback. 141568 parameters, 9 train
documents, 600 updates, 342462 effective targets and both seconds remain the
same original measurements. The historical original common.py fit_lm122–202
and _sync100–102 were re-read, as were new_lm45–47 and the necessary numerical
branches in historical text.run_text_foundation262–300. The original run.py
timing boundaries63–95 were also re-read. Those source hashes match the original
result's provenance. Main timing includes batch preparation/backward/clipping,
Adam updates, logging and periodic checkpoint saves; final NLL/final save follow
it. Whole-experiment timing includes the other comparison/evaluation/local-save
work. The initial exact label-count reconstruction is retained as initial
evidence rather than falsely described as fresh training or repeated GPU work.

Independent callback judgment: PASS. No substantive unresolved dependency.
New source bytes and the frozen whole-chapter input are explicitly distinguished
in current-input-provenance.json. Original issues/proofs and the original PASS
are preserved; canonical changes only add this owner's callback evidence and
the new terminology verification.
