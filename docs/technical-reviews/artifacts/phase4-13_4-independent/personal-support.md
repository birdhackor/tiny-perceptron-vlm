# Independent factual inspection of 13.4

Reviewer: `/root/phase4_factual_coordinator/factual_13_4`. Inspected on 2026-10-05.
This is a new technical inspection, not an inherited reader/technical verdict.
No previous report body or author correction summary was read.

Read the whole current 13.4, source lines 102–137, including the details block and
the original Python fence. Read 13.2 and 13.3 to establish complete answer-score
and EOS conventions, and 8.3 for the linked content checkpoint context. This is
not a chapter-first section, so an introduction inspection is not required.
`inputs/frozen-chapter-13.md` is the full chapter frozen input captured on first
reading, SHA c33e12dd7e61e73f11a45e3f7ee8c099dd130d23133c5a15426adfe2ca24adbe.
The formal source fingerprint is only `inputs/section.md`'s original UTF-8
bytes, SHA d041cdc0091f656d8e5f7c269c734145c20f24cf47c573e241166514df9ec3d4.

## Original authority personally inspected

- Rafailov et al., *Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model*, https://arxiv.org/pdf/2305.18290v3. The cached PDF's
  first page itself identifies arXiv:2305.18290v3, 29 Jul 2024. It is the authors'
  original paper, not a source-library synopsis. Personally read Eq. (7), §4
  “What does the DPO update do?” and “DPO outline”, and Appendix B's loss code
  and separate beta/learning-rate settings. Text extraction command actually
  used: `pdftotext -layout original/dpo-2305.18290v3.pdf original/dpo-2305.18290v3.txt`
  from this artifact directory. Eq. (7) uses beta times the difference of
  policy/reference chosen/rejected log ratios. Rewriting it gives beta times
  `(policy_chosen-policy_rejected)-(reference_chosen-reference_rejected)`.
  The outline uses the SFT policy as the reference whenever available and
  optimizes the policy for a given reference. The objective uses preference
  labels; it does not turn the reference model into a correctness oracle.
- CPython official v3.13.5 source,
  https://raw.githubusercontent.com/python/cpython/v3.13.5/Lib/copy.py,
  personally fetched over HTTPS and read lines 1–45 and `deepcopy` lines
  119–164. Generic deep copying makes a compound copy recursively while
  allowing per-type copying behavior. TinyLM storage independence was verified
  by execution, rather than assumed for every arbitrary Python object.
- PyTorch original immutable commit
  `5c4886908584029761b579af026dcfb627c84070`,
  https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/module.py,
  personally read lines 2894–2959: `eval()` is `train(False)` and
  `requires_grad_` mutates every parameter's flag. Same installed git revision
  as the CPU execution environment.
- Same PyTorch commit's original autograd documentation,
  https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/docs/source/notes/autograd.md,
  personally read lines 184–230 and 331–347. Grad recording requires a tensor
  input with `requires_grad=True`, and eval mode is orthogonal to autograd.
  The section's integer token inputs and frozen parameters produce no graph;
  eval alone does not provide this guarantee.

## Repository methods and raw measurements personally inspected

AST located function names/ranges before reading implementations. Read current
`tiny_perceptron/model.py` lines 1–89, alignment.py lines 29–42, data.py lines
1–28; read behavior.py `_preference_parts`, `_pair_examples`, `_dpo_train`,
`_preference_evaluate`, `_state_digest`, and the operational start of `run_dpo`
and `run_style`. Excluded the unrelated pilot/format-only branches and the
return `scope` explanations. The ordinary metric note describes the sum/EOS
contract and was not an author correction summary.

Retrieved original source with `git show <result-revision>:<path>` and checked
its complete bytes against the raw result's `/code_sha256`. DPO revision
`8a7571847dd0106c20aaeb8fe32c9d4a0e675c0d` contains behavior.py SHA
94ab92aa1b8017edbb8f5524ee598dce93756cf295377ac21515bf3c0a05bdc0,
common.py SHA df08fb4d378170e930bb5deb50b97749c33e5d97d0fbd3ce77ba916b12b07206,
and text.py SHA 04b0a75d151b6920dd8c28b5f7b6cadf94d27a71776a7ca7952be39ae05a0b59.
Read original behavior.py `run_dpo` lines 697–718, `_preference_evaluate`
609–641; original common.py `evaluate_lm` 243–304 and `split_records` 50–70;
original text.py `_save_splits` 47–62 and hash helper 39–44. For the content
origin, personally read original style revision
`ae7bbbf95537d228a44810041d2a9e978360d369` behavior.py lines 95–101,
matching its raw result's code SHA. Original methods establish that content.pt
was trained using SFT, DPO loads that dependency, each beta run deep-copies the
same base, the reference is excluded from the policy optimizer, and the run
checks numerical reference-state hashes before returning unchanged=True.

Before any raw result values were printed, inspected required top-level keys
and their types, then the named measurement subtrees' keys/types. Only the
measurement/provenance pointers enumerated in `measurements.json` and
`review_checks.py` were inspected. Full raw originals are retained, including
unread extra notes/scope; they have not been edited to manufacture inputs.
DPO original JSON SHA f298491097b9dcd7f62b2619804e744b8ad660a5831f5bfccbf5f2eedc947735;
style original JSON SHA b534bf8062231c452c01b259ff3419df53e1da771ec19c339549ed736f0f4384.

## Independent results and meaning

The original fence ran unchanged and printed True/False. The exact assignment
exercise ran with only `copy.deepcopy(policy)` replaced by `policy` and printed
False/False: storage/object aliasing freezes the policy too, and a manual
no_grad edit still changes the shared weights. Eval-only retained autograd=True.
No backward, optimizer update, preference training, or existing checkpoint
inference occurred. These observations support the example's copying/freezing
contract, not a claim that the random model learned preferences. Other fixed
reference implementations, such as precomputed scores, are outside this code
example's scope.

Arithmetic gives -1 for the reference gap, 0 for the policy gap and relative
improvement 1; equal -3 log probabilities are equal probabilities. DPO's
executed loss at beta .1 is 0.6443966600735709, agreeing with log(1+exp(-.1));
identical initial policy/reference gives margin 0 and log(2).

Reconstructed the original pure arithmetic/split functions with seed 42, not
training. The resulting 49/8/7 records and 28/4/4 families match every original
JSONL split SHA for both content arithmetic and DPO pairs. `3+5=?` is validation
sample index 2, with chosen 8 and rejected 9. Each answer has 2 scoring targets
(digit and EOS). Raw scores rederive reference -10.893460392951965, updated
policy -8.447893142700195, relative improvement 2.44556725025177. Rounded to
five decimal places these are -10.89346, -8.44789, 2.44557. The updated policy
gap remains negative; monotonicity of exp means rejected has greater sequence
probability. All 15 original held-out rows have initial relative margin zero;
each beta run's 15 recorded reference margins are identical to the baseline.

Rechecked all seven content test prompts' true arithmetic answers and raw token
exact-match criteria (answer before EOS; UTF-8 byte IDs shifted by eight), not
only their recorded booleans. All seven are wrong, confirming 0/7. Recorded
reference-state digest is
4cf7b1e2182ad1edcc028480246baf2fe55d1fe54576c532c63505071b658d98;
unchanged=True is supported by the original run's explicit before/after hash
guard and by invariant recorded reference margins. I did not load weights or
independently rerun that original state's hash computation.

## Execution record and limits

Final `execution.json` records the actual argv, cwd, exit status, time, code and
stdout hashes. Final CPU run completed in about 2.58 seconds, exit 0, with Python
3.13.5 and PyTorch 2.14.1+cpu, git revision above, CUDA unavailable. Original
GPU measurements remain original data (Python 3.13.3; PyTorch 2.14.1+cu126),
not a new training or evaluation run.

First two checker-script runs failed because my initial instrumentation assumed
the wrong split-hash convention and assumed saved sample messages included the
answer. Reading `_save_splits` and `evaluate_lm` corrected those assumptions;
failed code/stdout/stderr/receipts remain under `initial-run` and `second-run`.
The third run passed; its versioned code was reconstructed from the final patch
and verified exactly against that run's recorded code SHA, then preserved under
`third-run`. Final run adds checks across both beta branches and both held-out
splits. These are reviewer tooling failures, not textbook defects.

There are no figure/image references in this section. Its scalar arithmetic and
aliasing example do not require visual measurement, so no rendering/viewing is
claimed. No downloads of training data, model preparation, model weights,
full training, GPU reruns, textbook/figure edits, or commits were performed.
The empirical support is confined to these recorded toy arithmetic conditions;
the section itself correctly avoids promising truth from relative improvement.
