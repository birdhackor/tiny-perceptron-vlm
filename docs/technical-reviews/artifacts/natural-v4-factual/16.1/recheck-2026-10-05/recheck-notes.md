Same-owner factual recheck of 16.1, 2026-10-05

Reviewer: /root/v4_review_coordinator/factual_v4_16_1.
Before making any report change, copied my exact previous canonical report,
original raw section, introduction and T.8 prerequisite to this directory.
Previous report SHA: 42a32c90c7b74a5e42ab5ffd38a952db0594d6e160fcb2c9c86fddfde9ff569a.

Personally reread the COMPLETE current raw16.1 and COMPLETE chapter introduction.
Current section SHA: dcdf00e0de3a37f8f323b9677119b2cfbfab4400a07e6c421ffce4d9ddba8340.
Current introduction SHA: a8b2dae29d5488abdc4432aa08f634efdb946ae3b3d9b0a22045657c85ba6a78.
The only16.1 byte change is replacing ../training.md#T.8 with the direct
../training.md#選讀效率實驗的量測條件 fragment. Its code, numerical claims,
measurement explanation and complete introduction are otherwise byte-identical.

My newly written introduction understanding: the opening separates several
possible causes of slowness or large memory use. It first asks the reader to
locate cost, then introduces cache, packing, accumulation, precision, attention
blocking, backward recomputation and compilation as possible tradeoffs. CPU
examples establish inspectable computational relationships; real time and peak
memory gains must be measured under specified hardware, length and mode.
This overview does not claim every listed option is an unconditional speedup.

Personally reread the complete current T.8 efficiency destination, including
its FP32/TF32-off L4 setup, SFT starting model, the six separately reset training
branches, absence of branch memory fields for padded/packed, CUDA allocated
tensor scope, reserved-unused/driver/subprocess exclusions, inference three
warmups/nine measurements, training first-three exclusion, Flash component
scope and precision-starting-model distinctions. The actual destination has
no embedded SVG; its links to later numerical discussions were not silently
treated as figures already viewed. My necessary residual and dispatch-example
SVGs were previously rendered and personally viewed; current original SVG bytes
and those renders both match my earlier hashes.

Full T.8 byte comparison also found the official docs URL changed from
generated/torch.cuda.max_memory_allocated.html to
generated/torch.cuda.memory.max_memory_allocated.html. This is recorded rather
than calling the H3 change the only T.8 edit. After exactly the H3 promotion
and that URL replacement, the complete T.8 bytes equal the current section.
All substantive method text, code and numbers remain unchanged.

Actual browser navigation: my first guess /chapters/16.html had no method link;
a subsequent concrete GET returned404. This was a wrong publisher route.
The correct numbered page /16.1.html returned200. I actually clicked its sole
T.8的效率量測方法 link; Chromium arrived at
training.html#選讀效率實驗的量測條件 and found the actual H3, whose top was
73.578125 pixels in the viewport. I personally viewed both source-link and
destination screenshots. The target shows the relevant allocation exclusions
and timing scopes directly, without requiring the reader to search T.8.

The newly linked external PyTorch docs URL returned403 in this environment.
I did not claim to read that blocked page. Instead I reread the already
retrieved byte-identical original PyTorch v2.14.1 torch/cuda/memory.py API at
lines542-560 and torch/cuda/__init__.py reexport at1732. The installed2.14.1+cpu
exports torch.cuda.memory.max_memory_allocated and torch.cuda.max_memory_allocated
as the same function object in module torch.cuda.memory. This verifies the
namespace/API meaning without calling GPU memory functions or reporting HTTP
success. The original official source remains the factual authority.

43 earlier evidence/source/dependency checks all matched: previous artifacts,
registered repository sources, necessary prerequisite section bodies, external
original authority bytes, and personally viewed prerequisite SVGs. The earlier
CPU profiling, arithmetic, timing observations and original GPU record audits
are reused honestly with their original dates, configurations and scope.
No CPU performance rerun was needed for the navigation-only correction; no GPU
or full-book work was run. The original11 factual claim judgments are retained.

The original limit remains: GPU results were audited from original records and
matching run source, not reproduced. Per-update raw training times are absent
from the public exports, so their numeric medians were not recomputed. No
unresolved factual issue was introduced by the changed navigation target.
