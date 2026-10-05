# 17.10 independent factual inspection

Reviewer: `/root/phase4_factual_coordinator/factual_17_10`; access date: 2026-10-05.

Read current section 17.10 in full, course/chapters/17.md lines 304–343, and necessary
preceding section 17.9 (lines 262–303). Read factual-reviewer-instructions.md,
checker schema, section_facts.py interface/worker restrictions, clear-tutorial SKILL.md,
and review-protocol.md. No old technical/reader report, dispatch conclusion, author
results commentary or correction summary was read. Ordinary implementation docstrings
and method limitations were read as method contracts and independently checked.

The exact original UTF-8 section SHA is
`0b74789196ec746cfffed1cd4a54c710bb239c4fa8daacd32bd6f306bf7cf52d`.
The frozen whole-chapter SHA `758c083b1b3e83c698536b09e48f85326d6376ff807d136bcf82345656fe82fa`
identifies `frozen-chapter-17.md`, not a promise that another section remains unchanged.

Read `tiny_perceptron/quantization.py` lines 1–81, especially per-output-channel
maxabs scaling (8–17), actual packing/unpacking (29–44), QuantizedLinear (47–64).
The original section fence ran with section_facts.py and exited 0. Its stdout and
environment are retained. The bounded independent cpu_verify.py runs the 4/8-bit
variants and intercepts F.linear's actual complete reconstructed weight argument;
FP64 input demonstrates that the cited FP32 result is scoped to the shown input.

Read only top-level key/type lists of the original raw L4 JSON before selecting
measurements. Actual value pointers inspected:

- `/revision`, `/device`, `/seed`, `/torch_version`, `/python_version`, `/gpu`,
  `/peak_allocated_bytes`, `/code_sha256` (file fingerprints), and artifact path/hash/bytes metadata.
- `/results/runs/{fp32,packed4,packed8}/timing/{measured,repetitions,prompt_tokens,decode_tokens,prefill_seconds,decode_seconds,decode_seconds_per_token,cuda_allocated_before,cuda_peak_allocated,cuda_additional_peak_bytes}`.

No raw JSON `note`, limitations results interpretation, status judgment or author
summary values were consulted. The complete original JSON remains intact in its
permanent copy; original and copy SHA independently agree. Top keys/types of a
release manifest were seen only to locate inputs; no release judgments were read.

The original raw report revision is `5af615e5d7c9642afee800390fa072257f895d0c`.
`git show` recovered original compression.py and run.py. Their full SHA values
equal the raw report `/code_sha256` entries. Current full-file fingerprints differ,
but `_sync`, `_timing` and `run_quantization` source segments are identical. Actual
read locations of historical compression: 36–40, 178–212, 284–333, 449–495;
historical run: 79–90, 128–145. Current compression was also read at corresponding
timing/variant locations. AST function maps were used before those reads.

The original `_timing` is decorated with `@torch.no_grad()`, runs one warmup,
three prefill calls and 24 decode calls, and performs complete-sequence model
calls of lengths 47–54 during each eight-step generation loop. There is no EOS
early stop or KV-cache path in this helper. A bounded CPU ProbeModel executed the
original helper AST (including decorator); it verifies call lengths, divisors,
eval/no_grad and context-length guard. This is a method-contract check, not a
new model latency or quality result. `_packed` reloads its saved quantized payload;
all variants remain in the `variants` dictionary while probing. Each CUDA probe
resets the same peak counter, explaining the outer final peak's narrower scope.

Authority sources personally inspected:

1. https://arxiv.org/abs/2210.17323v2, original author paper, v2 (22 Mar 2023).
   Independently ran pdftotext -layout on original PDF; output SHA exactly equals
   cached extraction. Read PDF/text first page and sections 3 (Eq. 1 and OBQ
   correction, text 182–213), Practical Speedups (520–554), limitations (636–642),
   Appendix A.2.2 (839–865). Supports dynamic dequantizing quantized-matrix /
   full-precision-vector kernel, reduced memory movement rather than integer-only
   multiplication, hardware and workload limits. Its algorithm corrects weights;
   this repository's elementary round/clamp layer does not implement that objective.
2. https://arxiv.org/abs/1712.05877v1, original Google authors' paper, v1
   (15 Dec 2017; later CVPR 2018 paper). Independently ran pdftotext on PDF;
   output SHA exactly agrees. Read identity/header and section 2.2, Eq. 2–6
   (text 119–188), showing an integer-arithmetic matrix multiplication route and
   hardware-dependent fixed-point implementation. This is supporting background,
   not a claim that our reference code runs this route.
3. https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/linear.py.
   Independently fetched official exact installed PyTorch 2.14.1 commit; read
   Linear lines 53–134 (weight [out,in], bias [out], leading input axes preserved,
   F.linear call). Exact downloaded file is preserved and hashed.
4. https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_tensor_docs.py.
   Independently fetched exact official commit; personally read element_size
   1728–1743, numel 3622–3629, nbytes 6760–6767. Supports ordinary element
   byte accounting; this is numerical payload accounting, not serialization
   metadata, allocator overhead, or total resident RAM.
5. https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/utils/benchmark/utils/timer.py.
   Independently fetched exact official commit; read Timer lines 65–101 and
   synchronizing timer lines 19–20. Supports warmups, comparable inputs/runtime
   settings, accelerator synchronization and repeated measurements. CPU checks
   use installed 2.14.1+cpu; original L4 report used 2.14.1+cu126.
6. https://docs.pytorch.org/docs/2.14/notes/cuda.html. Personally read original
   HTML by independent HTMLParser extraction and matching content, plus text
   lines 621–645 (synchronization/timing), 758–772 (allocator management).
   Original HTML and text hashes were checked on their permanent copies.
7. https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/cuda/memory.py.
   Official exact version cache personally read at reset_peak_memory_stats
   381–397, memory_allocated 525–539, max_memory_allocated 542–560,
   memory_reserved 563–575. The implementation separates allocated/reserved
   counters and explicitly mentions GPU context outside tensor accounting.

For additional original source context (not needed as sole formal evidence), read
the original PyTorch 2.14 Understanding CUDA Memory Usage text 3760–3783 and
max/reset-peak official pages 479–497; read torchao official fixed commit
9cd4107562b35cb5e3c71fda77af0f5c7af90c3f int4_tensor.py dispatch 190–230.
Only raw source URL/version/cache fields of the immutable locator libraries were
used, never their claims_supported/limits judgments. AST/file-list search outputs
were used to locate sources, not as evidence of correctness.

No figures are referenced by this section. Numerical storage/axes and runtime flow
can be directly checked with the shown code; no image rendering was required.
No browser page layout or mobile readability audit was performed in this factual
review. No GPU run, training, data/model download, weights load/copy/save,
model-quality reevaluation or new authoritative benchmark score was performed.
The raw JSON contains aggregate times, not individual repetition samples; the
check verifies units, reported aggregates and helper averaging contract, not
reconstruction of unavailable per-repetition time records or statistical variance.
