# Actual independent review scope

Reviewer task: /root/phase4_factual_coordinator/factual_16_6.
Date: 2026-10-05 UTC. Technical review only, fresh single-section identity.
No old technical or reader report body, author review, result interpretation,
or correction summary was read. The existing canonical report was not opened.
No subagents, textbook or figure edits, commits, pushes, model/data downloads,
GPU execution, existing training reruns, or complete-model reevaluations occurred.

Read: factual-reviewer-instructions.md, checker schema, section_facts.py,
clear-tutorial SKILL.md and review-protocol.md; cloud onboarding skill and
onboarding reference to establish reuse of the existing CPU .venv.
Read all source bytes of section 16.6 and its sole Python fence. For required
background read sections 16.1 (units/allocator scope), 7.3 (answer targets),
7.7 (PAD/ignored targets), and W.6 (gradient meaning). Saved each original
section. The full chapter file was copied without interpreting its other
sections and is explicitly a frozen input snapshot, not a current full-chapter
approval. Chapter introduction was not reviewed because this is section 16.6.

No image, SVG, animation, or screen claim appears in 16.6. Its input values,
one-dimensional slicing, and example weights are displayed as numbers and
explained in prose; a spatial visualization is not required to verify these
claims. No render or desktop/mobile page review was performed. Figures linked
inside background sections 7.7 and W.6 were outside this single-section visual
scope; the review does not approve those figures or those sections.

Primary JSON discipline: inspected top-level keys/types, then nested keys/types
at /results, /results/accumulation, /results/runtime, /results/dataset and
/results/update_variants/{ordinary,accumulated}/{training,heldout/test}.
Values read were measurement/sample/provenance fields. cpu-results.json records
the exact pointers used by the substantive audit. Manual inspection additionally
read /results/update_variants/{ordinary,accumulated}/training/final to distinguish
training NLL from the reported held-out test NLL, /artifacts/{0..12}/{path,bytes,sha256},
and /code_sha256 for architecture.py, common.py, data.py and model.py.
Keys/types alone were read from /public_exports and unrelated variant containers.
No gqa_note/timing_note values or other author result commentary were read.
The complete untouched raw file, including uninspected values, is retained with
its full fingerprint as primary/efficiency-original.json.

Release locator JSON was used only to inspect top-level key/types,
/private_source key/types and /files/{0..7}/path and /files/{0..7}/sha256.
No reviewed/approved/approval/model-card/inference values were read, copied,
or used for judgment. No checkpoint weights were loaded or saved.
The immutable measurement revision and /code_sha256 led to git show
48a4f3e912b483d70aee57c42c2aac226534a9a6:scripts/course_experiments/architecture.py.
Its preserved SHA exactly matches the measurement's recorded code hash.

Code discipline: AST first located functions and classes. Read calculation and
method contracts only; primary/original-code-inspection.json and
primary/repository-code-inspected-locators.json list exact ranges and snapshots.
Current architecture.py was read at _train, _accumulation_probe, _mechanism_examples,
_sft_dataset, _text_dataset, _nll, _forward, _gradients, _gradient_error, and
run_efficiency's computation before its return block. Original revision functions
were then personally read at the saved original ranges. The _train memory_note
is a general CUDA allocator measurement domain and CPU-method limitation;
it contains no measured winner/loser or prior review verdict. Relevant original
and current method ASTs were compared by the CPU verifier and were identical.
common.py, data.py and model.py full fingerprints also match the recorded run.

Official authority inspection: direct docs.pytorch.org stable requests for
Tensor.backward, AMP examples, CrossEntropyLoss, BatchNorm1d, Dropout,
clip_grad_norm_, Optimizer.zero_grad and Optimizer.step returned HTTP 403.
No request was retried unchanged. Official GitHub raw sources were retrieved
at installed torch git revision 5c4886908584029761b579af026dcfb627c84070.
The old AMP .rst path returned 404; the same-revision .md path succeeded.
Personally read preserved original snippets and their actual line ranges:
backward leaf accumulation; tensor construction, square, mean, sum and item;
CrossEntropyLoss class-index ignore/reduction formulas; BatchNorm1d batch statistics;
Dropout training masks per forward call; global gradient clipping;
zero_grad and the implemented Optimizer.step docstring; AMP accumulation order;
CUDA memory_allocated/max_memory_allocated/reset_peak_memory_stats.
The initial AST selection for Optimizer.step selected an overload stub; it was
replaced with the actual implementation at lines 1117-1124 before any supporting
claim was recorded. Python's original 3.13 common-sequence documentation was
also personally read, especially Notes 3-4 (zero origin, omitted endpoints,
and i<=k<j). Full response hashes and retained excerpt hashes are recorded
in the primary retrieval metadata. No search summary was used as authority.

Execution: the first manual worker attempt used .venv/bin/python from /tmp
and failed with exit 127 because that relative executable path did not exist.
No code ran in that attempt. The actual section_facts.execute entry point was
then called from the repository with a 60-second timeout and exited 0; it saved
original stdout, stderr, environment, and command metadata. The independent
verify_cpu.py, also bounded to 60 seconds, exited 0 and saved code, actual command,
stdout/stderr, environment and results. Its toy SGD update is a bounded mechanism
demonstration only. It does not establish a trained language-model score.

Existing evidence limits: primary GPU run was NVIDIA L4, torch 2.14.1+cu126,
Python 3.13.3, seed 42; new CPU checks use torch 2.14.1+cpu, Python 3.13.5.
Saved gradient/weight maxima, update-token totals, timing medians and allocator
peaks were audited as original measured aggregates. Test token identities,
5/10 exact counts, EOS counts, 69 valid test labels, NLL quotient, and ms/MiB
conversions were independently recomputed from raw records/measurements. Per-step
latency samples, training sampled batch identities and checkpoint states were
not reconstructed; no new numerical model score was substituted for raw evidence.
