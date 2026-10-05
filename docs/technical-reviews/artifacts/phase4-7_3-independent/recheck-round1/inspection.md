# Actual full reread of revised 7.3 — 2026-10-05

Same factual reviewer: `/root/phase4_factual_coordinator/factual_7_3`. This is my own recheck after the coordinator changed the manuscript; the coordinator's stated outcome was not used as a judgment. I read all present 7.3, including unchanged code, explanations, exercises and closing details, in one actual read. I reread the actual `render_chat`/ByteTokenizer/shifted contract, TinyLM embedding/forward and loss contract, and causal attention-mask code. I also reread the downloaded original official Transformersv4.57.1 code lines1061–1064 and1277–1282, PyTorch2.11 class-index CE equations/ignore_index, and TRLv0.24.0 assistant-only objective/template restriction. The original broad source inspection remains documented separately; I did not refetch or claim rereading every unrelated API page in this recheck.

Original source SHA: `4d946d60a534462cfc408b970b7cb8d0d0ab8ed3b66231f2c061fe4a92dcc154`.
Current full section SHA: `5c36cbb802c83d3e23586457498db50d0763152681d26a359cd20baca0b313c2`.
Own initial revise history: `docs/technical-reviews/history/phase4-7_3-own-initial-revise-f5959d637d2a809131856bd91da08ebf5f8c6526e3f7e67c98caae7294ee4c46.json`, SHA `f5959d637d2a809131856bd91da08ebf5f8c6526e3f7e67c98caae7294ee4c46`. This is byte-identical to my permanently preserved `initial-review.json`; neither was edited.

## Whole-section reread and substantive judgment

- Opening paragraph: keeping the question in X while supervising response content/end remains a correct distinction between context and next-token targets.
- Former line76 now reads “因X每格預測下一項，答案與是否計分的標記要一起配到前一格的預測位置”. This explicitly gives the actual original-to-predictor mapping. I checked `return tensor(ids[:-1]), tensor(targets[1:])`, the role-target construction, and the official source slicing against it. No second shift is required by TinyLM's inspected loss.
- I reread the whole unchanged Python fence: `render_chat` builds the pair; `enumerate(zip(..., strict=True))` prints corresponding X/Y positions; status follows label!=-100; sum/item count effective targets. It does not call a model, backward, optimizer or evaluation. The short trace is sufficient to expose each index; there is no figure reference and no image-render claim.
- Former line93 now says “程序逐位置列出當前輸入ID、下一項的label與是否計代價”. This removes the wrong/ambiguous direction word while retaining the genuine next-token relationship. Three answer bytes plus response EOS still give four targets. Role marker **targets** are ignored, but an assistant marker **input/predictor** slot can receive the first answer target; the text is consistent with that distinction.
- Conditional/status explanation and Boolean tensor count are unchanged and remain supported by the exact executed original trace and API inspection.
- Embedding X IDs versus loss Y/-100 paragraph remains correct. Ignoring direct user targets leaves context available under causal attention; it does not assert universal nonzero gradients or that masking alone yields good answers. The demonstration-quality reminder avoids using format correctness as answer-quality evidence.
- Exercises remain exactly `答案`:6 UTF-8 bytes+EOS=7, and `OK`:2 bytes+EOS=3, with the original three question bytes retained in X. Byte unit, not visual character count, remains explicit.
- Closing details only link longer recipes and preserve scope; no long training command occurs in this section. No empirical model metric or model-performance assertion is introduced.

I found no new substantive uncertainty or error. The original direction issue is resolved in the current manuscript. The original claim, its contradicted status, both quoted phrases, exact7–10→6–9 counterexample and pending initial verdict remain preserved in the initial history and current report's prior-version record. The current claim is separately verified for the corrected sentence; the old claim is not retroactively certified.

## Actual new execution versus reuse

I newly ran extraction only:
`.venv/bin/python docs/review-tools/section_facts.py 'course/chapters/07.md#7.3' --output outputs/phase4-7_3-independent-recheck1`
Exit0; extraction says `execution_exit_code:null`. Its current raw section, exact fence, metadata and BOOTSTRAP were copied here permanently. No original-fence rerun is claimed.

I newly ran the bounded script:
`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 30 .venv/bin/python docs/technical-reviews/artifacts/phase4-7_3-independent/recheck-round1/recheck_facts.py > docs/technical-reviews/artifacts/phase4-7_3-independent/recheck-round1/stdout.txt 2> docs/technical-reviews/artifacts/phase4-7_3-independent/recheck-round1/stderr.txt`
Actual exit0, unified exec session28280. This executes the current `render_chat` and hashes/compares bytes; it does **not** rerun the prior variants, loss, model forward/backward or gradient proof. Current environment: Python3.13.5, torch2.14.1+cpu git5c4886908584029761b579af026dcfb627c84070, CPU, CUDA build None, one thread, bash login:false, repository cwd.

The current newly executed alignment is exactly:
full IDs `[1,3,237,157,151,2,4,239,181,156,2]`, full targets `[-100,-100,-100,-100,-100,-100,-100,239,181,156,2]`;
original valid target indices `[7,8,9,10]`; current valid Y/predictor indices `[6,7,8,9]`.
X6 is assistant4; Y6 is first answer byte239. Thus the label and validity mask both take the original next item `targets[i+1]` once. Exact integers/list equality, no float tolerance needed for this new alignment check.

The new fence SHA `78601d85d5fae0b735c4a33d8e5561f4b4a38afc9f01ebda7a330e4685b58249` is byte-identical to the executed initial fence; BOOTSTRAP is byte-identical. Current data.py/model.py/attention.py/build_course.py/section_facts.py/check_technical_reviews.py match all corresponding original snapshots exactly. Every one of the46 prior report artifact hashes matches, including all official snapshots and actual original/probe outputs. The fresh alignment X/Y equals the initial probe's original-example X/Y. Therefore I **reuse** the previously personally executed4/7/3/1 counts, role/boundary, denominator and gradient evidence for unchanged code; I do not label those checks as new runs.

`section.diff` is the actual full raw-section textual diff: only the two stated phrases changed. `facts.json`, stdout, stderr, code, metadata and current bytes are retained in this directory. No body/figure edits, training, existing-model evaluation, GPU, course-data preparation, data/model download, model checkpoint or commit were performed by this reviewer.

## Figure and introduction applicability

Current extraction still has no SVG or figure reference; figure_sha256 remains `{}`. The diagram requirement is not applicable, and no adjacent7.4 figure is certified. This is not a chapter-first section; introduction review remains not applicable. The section recheck does not substitute for the coordinator's separate reader recheck.
