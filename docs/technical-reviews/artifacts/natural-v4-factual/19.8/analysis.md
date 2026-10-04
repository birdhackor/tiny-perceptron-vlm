# Independent factual review of current 19.8

Reviewer: `/root/v4_review_coordinator/factual_v4_19_8` (fresh context).
The 7,189 original UTF-8 section bytes, including blank lines, hash to
`c758c8aeadc22d10bf2627a341a08a14a1c2412f75b71dbc78e5d2a7b442437a`.
The previous assigned report was copied without opening it. No donor reports,
author checks or prior verdicts were consulted. This section has no direct SVG;
the necessary 7.18 prerequisite figure was inspected with its actual current bytes.

## Original authorities actually read

Original PDF URLs, byte SHA-256, sizes and retrieval receipts are in
`retrieval-receipt.json`; complete originals are only in ignored research storage.

* DPO, arXiv:2305.18290v3, 29 July 2024: PDF pages 3–5, §3 and §4,
  equations (3) and (7), and the gradient/DPO outline. Equation (7) directly
  compares the preferred/dispreferred policy log-probability difference with the
  same fixed reference difference. It avoids a separate reward network and RL
  sampling loop. Its preference labels specify what is favored; they do not
  independently establish factual correctness. The derivation is for a reference
  regularized preference objective, not a guarantee of identical trained weights
  under an arbitrary PPO implementation or of improved open-ended answers.
* InstructGPT, arXiv:2203.02155v1, 4 March 2022: PDF page 6, §3.1,
  all three steps. Human labelers supply demonstrations and comparisons; a reward
  model learns comparisons and PPO then optimizes the policy. RLHF names this
  human-feedback provenance/process, while PPO names an update algorithm.
* PPO, arXiv:1707.06347v2, 28 August 2017: PDF pages 1–3 and 5,
  abstract, §§2–3 equations (1), (6), (7), and algorithm 1. Old-policy probabilities
  belong to the current rollout and are held fixed across its update epochs;
  these are distinct from a post-training reference retained across rollouts.
  PPO is an RL update method, not a required stage before or after DPO.
* DeepSeek-R1, arXiv:2501.12948v1, 22 January 2025: PDF page 5,
  §§2.1–2.2, and pages 9–10, §§2.3.1–2.3.2. R1-Zero starts RL from a base model
  without SFT cold start; R1 uses cold-start demonstrations before RL. These are
  concrete counterexamples to a universal mandatory SFT/RL/PPO/DPO ordering.
  No small capstone reproduction of their reasoning results is inferred.
* Mixtral, arXiv:2401.04088v1, 8 January 2024: PDF pages 2–3,
  §2.1, sparse weighted expert sum and top-K router definition. It supports sparse
  expert selection, with shared model components remaining outside expert choice.
  The capstone uses four experts/top-2 rather than Mixtral's eight/top-2.
* Switch Transformers, JMLR 23(120), April 2022: PDF page 6,
  §2.2 “A Differentiable Load Balancing Loss,” equations (4)–(6).
  An auxiliary router-probability/load term encourages balanced dispatch;
  it does not assign semantic expert specializations. The project normalizes
  dispatch over top-2 assignments and removes PAD positions; it is not a claim
  to reproduce Switch's top-1/capacity implementation.
* PyTorch original `torch/nn/modules/module.py`, commit
  `5c4886908584029761b579af026dcfb627c84070`, lines 2916–2955:
  `eval()` calls `train(False)` and `requires_grad_` changes every parameter's
  gradient flag. The fetched original is byte-identical to the installed source.
  The actual interpreter is Python 3.13.5 with Torch 2.14.1+cpu, no CUDA.

## Formula and bounded execution

Let C/R denote preferred/dispreferred complete answer probabilities, including
EOS and excluding prompt/PAD targets. The checked code implements
`m = (log πC − log πR) − (log refC − log refR)` and
`L = −log sigmoid(0.1 m)`. Identical deterministic policy/reference weights give
`m=0`, hence `L=log(2)=0.6931471805599453`, regardless of their original ordering
or unequal answer lengths. Observed FP32 `0.6931471824645996` differs by about
`1.90e-9`. Rounding to four decimals gives 0.6931.

The seed-42 first pair is the training `numbers:3:6` tool-return row, with
`DIRECT:9` and `DIRECT:9，祝你愉快！`. Its answer targets are 9 and 27 byte/EOS
positions. Independently summing the selected log-softmax entries agrees with
the implementation within 3e-5 (FP32 reduction order). Both initial margins are
103.11385345458984. This raw preference ordering is not training success.
All 92 synthetic pairs come from training families (44 tool-return, 24 style,
24 RAG); no held-out family is included.

The section demo has no checkpoint load, optimizer, backward or update.
The additional own probe performs exactly one random-CPU optimizer update on
two rows/two pairs, with the documented mixed coefficients, to check copying,
gradient isolation and optimizer membership. It does not test learned quality.
Reference state SHA stays equal before/after; policy SHA changes. The existing
reference-frozen and answer-mask tests actually ran: 2 passed, 27 deselected.

## Official run records independently audited

Each inspected official file was fetched from the section's pinned project commit
`1df335318bda03fd771807f66976953231d5a00b` and matched current repository bytes.
The official DPO report's full training-code hashes match current inspected code.
The DPO `train-report.json` exactly matches the wrapped result's `results` object.
Its parent SHA equals the joint inference export SHA, and its completed schedule
records 100 updates at seed 42 on NVIDIA L4, Torch 2.14.1+cu126.

The actual objective is `DPO(beta=.1) + .2 assistant-only CE + .01 auxiliary`.
Each saved history objective was independently reconstructed from its three terms;
the maximum discrepancy is 2.23e-8. First/last DPO terms are
0.6931471824645996 and 0.00004602140688803047, both before the respective update.
The seeded source reconstruction uses 24 replay rows and 12 preference pairs per
update, and reproduces 41,403 answer/EOS replay positions over 100 updates.
First and last preference draws differ. Reconstructed IDs are not a saved runtime
batch log, and the reported margins are averages; they do not permit independently
recomputing each pair's GPU loss. No before/after full reference-state fingerprints
are present in the inspected GPU reports. The section correctly limits that claim.

All joint/DPO validation generated byte IDs were decoded independently. Scoring
checks terminal EOS, exact action payload, ordered calculator arguments, runtime
sum and final generated content against the original split's expected answers.
Every per-row stored flag and aggregate matched. Both use the same 84 IDs/inputs,
with 3 style and 18 joint image/audio examples. Style stays 3/3; joint image/audio
changes 18/18 to 14/18; total changes 75/84 to 71/84. The four new failures are
joint examples; all nine shape failures remain. This supports the particular
joint-versus-DPO choice, not universal DPO harm or safety/style generalization.

Training reports mark final tests unevaluated. The official deployment record
names joint as recommended based on validation and records the recipe as fixed
before test. The frozen deployment source explicitly uses joint for both PTQ
exports, reloads them and evaluates them; their source SHA matches the joint
checkpoint. Independent per-row rescoring gives 78/90 for each PTQ export.
The student record completes both 350-update Dense branches; its teacher SHA
matches the DPO export, not joint. These are original run-record provenance
checks, not our own compression/student runs or an independent historical witness
to the original pre-test decision. The finite-PPO original record separately has
four precomputed candidate answers, 148 policy parameters and zero token-training
positions, consistent with the section's explicit conceptual side-branch limit.

## Personally inspected figure

Chromium did not finish its initial render and was stopped; its actual stderr is
retained. Inkscape then exited 0 and produced `posttrain_signals-current.png`,
which I personally viewed. The left demonstration card points down to SFT; the
middle preference card points down to DPO and additionally right/up into the
scoring card through “first teach scorer, then score new responses”; the right
scoring card points down to PPO or other updates. There is no SFT→PPO→DPO chain.
Both sample answers retain 3, and the footer distinguishes human feedback from
programmatic checking. Positions, arrows, numbers and labels agree with 7.18 and
19.8. Its current SHA is recorded separately, not inherited from an older review.

No source or checker changes, installation, model/new-training-data download, GPU execution,
publication or reading-time edits were performed. No unresolved substantive
claim or correction was found within this section's stated scopes.
