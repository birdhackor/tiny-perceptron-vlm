# Independent inspection of lesson 9.1

Reviewer: `/root/integration_technical_coordinator/fact_v2_09_01`, fresh context.
Date: 2026-10-04. No earlier review conclusions or author history were used.

The complete technical review guide and section 9.1 were read. Explicit prerequisites read:
`08.md#8.1`, `first-steps.md#W.2`, `06.md#6.1`, `06.md#6.6`, `01.md#1.8`, and
`docs/course-experiments/release-licenses.md`. This section and these prerequisites have no referenced SVG.

## Original sources inspected

- Ouyang et al., arXiv `2203.02155v1`, original PDF retrieved directly and converted with
  `pdftotext -layout`. Introduction (p. 2), §3.6 (pp. 9–10), and Appendix B labeler
  instructions (Appendix B.2, p. 36; Figure 10, p. 37) distinguish useful task
  assistance, non-fabrication, and harm.
  Appendix B explicitly supports clarifying confusing instructions and using only supplied
  information in tasks that require that. §3.6 cautions that observing true statements is
  not a measurement of a black-box model's internal beliefs. This lesson uses an explicitly
  task-specific operational standard; it does not establish psychological honesty.
- PKU-Alignment dataset README at fixed revision
  `9421ffafec3fa40a1f1a7d567b4d525079477ecb`, retrieved directly. YAML declares
  `cc-by-nc-4.0`. “Dataset Summary” and “Human-Preference on Harmlessness and Helpfulness”
  distinguish individual Q-A labels from two independent relative preferences. “Ranking
  of Responses” says harmless responses rank above harmful responses, but never says a
  winner of a both-harmful pair becomes harmless. The archived original viewer response
  has row indices 0–99, no truncated cells, and rows identical to the source JSONL.
- CC BY-NC 4.0 original English legalcode, §1(i), §2(a)(1), and §3(a): noncommercial
  reproduction/sharing and adapted material are permitted subject to conditions. The
  direct fresh request returned HTTP 403; the existing original HTML snapshot was read
  instead (not its surrounding retrieval notes or summaries). Private-only derived
  weights are project policy, not a blanket term in the license.
- CPython `v3.13.5` official reference sources: expressions “Subscriptions”; lexical
  analysis “f-strings”; Unicode HOWTO “Encodings”. Lookup evaluates its key expression;
  f-string braces contain an evaluated expression. UTF-8 encodes non-ASCII code points
  into multiple bytes. The example's different quotes are valid, including on older
  Python. Python 3.12+ also permits reusing a quote type in an f-string replacement;
  the lesson does not claim different quotes are mandatory on every version.
- PyTorch `v2.14.1` official `torch/nn/modules/loss.py`, `CrossEntropyLoss` documentation
  and class-index equation: negative log softmax of the correct class, masked at
  `ignore_index`, normalized over nonignored unweighted targets. This is predictive NLL,
  not a safety classifier or a count of incorrect complete responses.

## Numeric and software checks

`fact_v2_09_01_audit.py` runs the literal lesson Python block and its exercise on CPU.
It reconstructs all selected records from the SHA-verified public source archive,
checks every archive viewer row against the JSONL, and reconstructs exact formal split
file hashes. It checks masks, dtype (`int64` labels/input), byte budgets, EOS supervision,
family separation, and cumulative sampled effective targets. It also executes the
actual generation stopping code with fixed next-token logits and the current release
approval/rejection and redaction functions. Current `scripts/course_release.py` was
read completely (all 522 lines); relevant preflight/backup/public-projection code was
also inspected.

Hand derivation of the split: 60 distinct prompt families; `floor(0.8*60)=48`,
`floor(0.9*60)=54`; validation is `54-48=6`; test is `60-54=6`.
Each supervised example has `answer_UTF8_bytes + 1` effective labels, including EOS.
The user text and structural roles have label `-100`. Input length is
`prompt_bytes + answer_bytes + 4 <= 244 < 256`; padding is ignored in NLL.
The 48 unique training examples contain 5,599 targets; validation/test each contain
726. The seeded 80 updates of batch size 4 reuse examples and total 37,709 targets;
this is not 37,709 unique targets or 320 unique examples.

For the CE arithmetic check, logits `[0,2,0]` with target 1 give
`log(exp(2)+2)-2 = 0.2395447662218846`; an added ignored target does not change the
denominator of 1. The audit separately recomputes the new model's initial training NLL
on CPU: 5.757517037082515, agreeing with formal GPU 5.7575168626652085 within 2e-6.
Formal values rounded to five decimals are 5.75752 and 2.76322 (rounding tolerance
5e-6); they are full-training-dataset target-weighted NLL before and after training,
not the logged sampled batch losses.

## Complete formal result and limits

The original full result is `outputs/course-control/37046840520/result.json`; its full
SHA is preserved in the durable sanitized audit. All validation and test samples were
read and checked against rebuilt prompts and targets. Every one of 12 generated ID
sequences has length 128, no ID 2 (EOS), and differs from the expected answer IDs.
The audit preserves row identifiers, complete sequence fingerprints, and per-row
checks, without copying the private-branch text into this report.

Formal configuration: seed 42; TinyLM width 32, one layer, one head, vocab 264,
max context 256, 37,728 trainable parameters; 80 AdamW updates, batch size 4,
learning rate 0.003, answer/EOS-only labels, float32, deterministic greedy generation.
Formal environment: Python 3.13.3, PyTorch 2.14.1+cu126, NVIDIA L4. Pilot fit time
0.446771826 seconds covers update-loop work and intermediate checkpoint saves, but
excludes initial/final full-dataset NLL and the final checkpoint save. Full experiment
elapsed time 22.208399321 seconds includes toy branches, evaluation and local saves;
it excludes image build, startup and HF uploads. No throughput or GPU speed claim is
made in this lesson or inferred from the CPU audit.

An additional read-only CPU execution of the actual existing checkpoint completed
in GitHub Actions run [37173099328](https://github.com/birdhackor/tiny-perceptron-vlm/actions/runs/37173099328).
The reviewer wrote the probe independently before it was run. The controller read
the anchored `result.json` to check lineage, then streamed
`course-v1/safety/pku-pilot.pt` and the three `pku-excerpts/*.jsonl` files from the
existing Modal Volume; it started no Modal function/container and trained no model.
The receipt (`fact_v2_09_01_checkpoint_receipt.json`, SHA-256
`57f815dcd225e1ec14c163851e17701a38bb91edf2fd599fbf3b44b122528c52`) was then read
independently, and the local audit compares every file/code/probe hash and all 12
generation fingerprints against the original evidence. These comparisons passed.
Checkpoint size is 488,461 bytes, SHA-256
`8f5e7857c1d5422bbf3adcef95a9ecba8f598b8a33ad2e0d931e4f08a8155c3b`;
both saved step and AdamW optimizer step are 80. Remote CPU environment is Python
3.13.15, PyTorch 2.14.1+cpu, two threads. Final train NLL is 2.763223186478612
over 5,599 targets, versus formal GPU 2.763223360895919; difference 1.74417307e-7.
Validation/test CPU NLL are 2.7731440940835914 and 2.7767278497869317 over 726
targets each. All 12 CPU generated-ID fingerprints exactly equal the original GPU
fingerprints. No private weights or source text were returned to this workspace.

The raw run's preflight says the backup repo is private, and its HF backup reports
verified fixed checkpoint downloads. Its older `private_full_training_state` flag
is used only as evidence of full-directory backup, not exact resume. Public projection
of the original PKU branch matches the current public `safety.json` exactly. An
anonymous download of the pinned public export manifest shows only `safety-only.pt`
and `model.pt`, with source hashes matching the approved synthetic branch. Existing
public raw source archives and older public Actions diagnostics are documented in
release-licenses.md; this section's statement about the current organized report does
not erase those historical copies.

Cropping ends at complete Unicode code points, not necessarily grapheme clusters or
semantic boundaries. The truncation and head-100 sampling prevent general natural-
language safety conclusions. Separate labels and a falling training NLL alone do not
show useful, honest, or safe model behavior. No paid compute or training was started
by this reviewer.
