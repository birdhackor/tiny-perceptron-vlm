# 13.17 independent factual evidence

Reviewer: `/root/phase4_factual_coordinator/factual_13_17`, fresh context, 2026-10-05.

The reviewed section is the exact original UTF-8 byte range from `## 13.17` to the next `##`/EOF. Its SHA-256 is `8f8adcf0335aefec1c784d66ec8574d3ceab4b775149f8c43eaa84483860ebef`. `inputs/chapter13.frozen-input.md` is the **frozen first-read input snapshot**, SHA `cb2d732d16a9cace68059c76ed70671e83516f41cc7a253920764a2291dc1b86`; this fingerprint does not describe later edits to unrelated chapter sections. This is not the chapter's first section, so chapter-introduction review is not applicable.

Personally read the complete current 13.17 text, its SVG, prerequisite 13.3–13.6 and 13.16, factual instructions, checker schema, extraction helper, clear-tutorial skill and review protocol. Exact prerequisite bytes are saved separately. No previous review verdict, reader report, author correction summary, or inherited source inspection was used. Original papers were located with the supplied immutable locator index and then read directly. The only inspected result narrative fields were ordinary dataset/provenance conventions (`/revision_scope`, `/results/sft/demonstration_mode`); comparison summaries/limits, evidence/status/output-scope narratives were not inspected. Source method docstrings and input/output contracts were read because they define the experiment.

## Original authority, version and personal support

1. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, https://arxiv.org/pdf/2305.18290v3 . First page verifies arXiv v3, 29 July 2024, Stanford authors, NeurIPS 2023. Local original PDF SHA `92cb3a2b71362acda98a789b03d88688fd33cf5fcf13f81d2b1de30ee7d3b67a`. Read original pages 1–5 and theorem discussion on page 6 (local extracted text lines 1–335). Figure 1 and §3 define separate RM/RL and direct-policy pipelines; §3 Eq.3 and §4 Eq.4–7 derive the reference-constrained objective, log-ratio difference and logistic preference loss. §4's outline initializes the reference from SFT and optimizes an offline preference dataset. §5's equivalence is conditional on the modeled objective/preference assumptions: it does not assert equality of any two finite implementations' trained parameter arrays.
2. Schulman et al., *Proximal Policy Optimization Algorithms*, https://arxiv.org/pdf/1707.06347v2 . First page verifies arXiv v2, 28 August 2017, original OpenAI authors. PDF SHA `e78feadadbdbb0b601b3c2bcc81404722cd431a489b307545f9b7bea1e8c4f5b`. Read pages 1–5 (text lines 1–270): §3 Eq.7 is the minimum of clipped and unclipped probability-ratio surrogates; §5 Eq.9 includes the value-function squared error, and Algorithm 1 alternates current-policy data collection with reused-data optimization. These support the actor-critic route depicted here; PPO in general is not defined by a language-model reference policy.
3. Ouyang et al., *Training language models to follow instructions with human feedback*, https://arxiv.org/pdf/2203.02155v1 . First page verifies arXiv v1, 4 March 2022, original OpenAI authors. PDF SHA `c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98`. Read pages 1–3, §3.1 (text lines 295–319) and RL subsection (484–510). Figure 2 and §3.1 explicitly use human comparisons to train RM and use PPO on the supervised policy. The RL subsection uses an SFT KL penalty and a learned value function. Human labels are what makes this the cited human-feedback workflow; our synthetic fixed-rule cards and no token generation do not establish an LLM RLHF deployment.

Original text conversion commands were `pdftotext -layout <sources/name.original.pdf> <sources/name.original.txt>` for the three named PDFs. HTTPS URLs identify the original papers; these were read from their exact preserved PDF versions, not freshly downloaded or replaced by search snippets.

## Independent math and software

For preferred first card, uniform reference and beta=1:

`margin = ln(p_chosen) - ln(p_rejected)` and `sigmoid(margin) = p_chosen/(p_chosen+p_rejected) = p_chosen`.

Therefore `L = -ln(p_chosen)`. Units are dimensionless natural logs; both probabilities sum to one; one preference pair means `.mean()` divides by one.

| probabilities | relative margin | exact loss | 4 decimals |
|---|---:|---:|---:|
| 0.5 / 0.5 | 0 | 0.6931471805599453 | 0.6931 |
| 0.7 / 0.3 | 0.8472978603872037 | 0.35667494393873245 | 0.3567 |
| 0.3 / 0.7 | -0.8472978603872037 | 1.2039728043259361 | 1.2040 |

The original fence was genuinely run by the existing CPU/offline helper, exit 0, Python 3.13.5, torch 2.14.1+cpu. Its real stdout, stderr, environment and command receipt are saved in `execution/original-fence.*`. API group coverage: `torch.tensor` establishes a two-entry floating probability vector, `.log()` uses natural logs, `[:1]`/`[1:]` form matched one-pair vectors, `dpo_loss` subtracts the detached reference margin and takes mean negative `F.logsigmoid`, and `.item()`/`round(...,4)` report scalar values. No optimizer, parameterized network, backward or update appears in the original fence.

The own bounded audit runs all three probability cases in float64, checking equality within 1e-12, gradients in the expected chosen/rejected directions, zero reference gradients, three-pair arithmetic-mean reduction, cancellation of a shared reference log-score offset, and rejection of beta<=0. It never invokes `run_posttraining` or loads/saves model weights. Original float32 display values are independently confirmed by its real stdout. The audit code, real stdout, empty stderr, environment and actual command/exit receipt are retained permanently.

## Existing measurements, recomputed rather than retrained

Raw input is the intact `docs/course-experiments/results/posttraining.json`, SHA `95c8ee6c89b009a9646903c1bccaa6a0ace53786a608307d50ba7d9fb0a09172`. First inspected upper-level keys/types, then explicit provenance/config/count/time/sample pointers. The full list of 453 checked pointers is in the audit stdout; no recursive narrative dump was used.

Three original code hashes all exactly match the named current source and frozen copies. The recorded Git revision is the base checkout before the experiment was committed, as `/revision_scope` explicitly states. A `git show <revision>:scripts/course_experiments/posttraining.py` lookup actually failed with exit 128; this is preserved in `code/measurement-source-provenance.json`. Exact `code_sha256`, not that base tree, resolves the measurement code version. This corrects an early unverified suspicion of source mismatch.

Read original `build_records`, `split_records`, `_evaluate`, `run_posttraining` computation through its count/time/provenance fields, and model/helper contracts; exact read ranges are saved. Generated input records only (no training), reproduced all three dataset hashes, family sets, contexts and preference-pair counts. The unordered-operand representation only emits a<=b; the family explanation does not assert that swapped prompt duplicates were actually present. Each family retains all three modes in one split. Train contains 44 families/132 prompts/660 unique preference pairs, validation 5/15/75, test 6/18/90. No family is shared between splits.

All 18 test rows were checked against regenerated task inputs, labels and candidates; each recorded policy action was recomputed by argmax over its four stored probabilities, compared to independently specified mode→best-action mapping, and checked against recorded response and success. New totals:

| policy | number | explanation | missing quantity | full request |
|---|---:|---:|---:|---:|
| SFT | 6/6 | 0/6 | 0/6 | 6/18 |
| PPO | 6/6 | 0/6 | 6/6 | 12/18 |
| DPO | 6/6 | 0/6 | 6/6 | 12/18 |

Independent success means choosing the full-request card, not reward magnitude or deep explanation. The method's candidate sentence only contains operands and answer. The network selects four precomputed candidates from four structured features; it is not learning arithmetic or generating text.

PPO uses 120 rollout batches × 3 epochs = 360 policy and 360 critic updates, 120×64 = 7,680 newly drawn actions. DPO uses 360×64 = 23,040 preference-pair draws; repeated draws are not unique pairs. RM has 300 updates. SFT has 60 steps and number-only demonstrations. Thus a common 360-policy-update count does not give equal total work or a same-data/same-budget SFT control.

The saved policy architecture has `(4×16+16)+(16×4+4)=148` parameters. Original method uses two independent deep copies of the fixed SFT reference; recorded PPO/DPO initial and reference-before/after state hashes are identical. A bounded *untrained* copy contract checks every tensor is equal without claiming to have reloaded the historical checkpoint arrays.

Original single-run CPU segment times are PPO 0.6570014490000062 s→0.66, DPO 0.34335579199995436 s→0.34, RM 0.2704430690000095 s→0.27. These clocks surround the respective update loops; they are not end-to-end large-model benchmarks, replicated confidence intervals, or fresh timings of a retraining here.

## Figure and limits

Ran Inkscape 1.4 on the original SHA-pinned SVG, export exit 0, output 640×860; personally viewed the resulting PNG. The single SFT/reference box branches left to RM/policy/critic PPO operations and right to preference/reference DPO operations, then both feed independent task checks. There is no PPO→DPO arrow. Text and arrow endpoints are legible and not clipped. Native PNG and render receipt, including harmless renderer warnings, are saved. Chromium was not used; no browser-page or mobile layout certification is implied.

No data/model preparation, model download, full training, GPU work, source-text/figure changes, or commit occurred. No original weights were loaded or copied. These are one existing fixed CPU recipe plus bounded arithmetic/contract checks, not evidence of universally better alignment, a full LLM RLHF run, same-budget SFT superiority, or final-product integration. Lower DPO loss concerns the supplied preference labels; actual selected/generated outputs and retained abilities still require separate evaluation. All substantive claims checked within this scope support pass.
