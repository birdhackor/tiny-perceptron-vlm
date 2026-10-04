# Original-authority inspection notes

Reviewer `/root/v4_review_coordinator/factual_v4_9_8`; retrieved/read 2026-10-04.
Complete original documents remain only in ignored factual-research/9.8.
retrieval-receipts.json records the retrieved complete-document SHA-256 values.

* scikit-learn 1.8.0, original doc/modules/cross_validation.rst, lines 9–20 and
  60–82: fitting and testing on the same samples rewards repetition; test is
  held out, validation can be used for development, and model selection against
  test can invalidate generalization estimates. Lines 628–678, grouped-data and
  GroupKFold sections: related observations belong to domain-defined groups;
  groups shared between fitting and evaluation do not measure unseen-group
  generalization. This supports template grouping when the grouping actually
  captures dependency. It does not certify that the chapter's two example
  strings are distinct semantic families, nor prove broad model generalization.
* scikit-learn 1.8.0, original doc/common_pitfalls.rst, lines 77–119, Data leakage:
  the original instruction is "never call fit on the test data". This supports
  withholding test examples from parameter training and replacing a test set
  after it becomes part of training. Test-dependent development can also leak
  information without gradient updates; frozen weights alone are not sufficient
  to certify independence of every future evaluation.
* CPython v3.13.5, Doc/library/stdtypes.rst, dict.items lines 4795–4798 and
  Dictionary view objects lines 4900–4931: items returns (key, value) pairs,
  and views iterate in insertion order. The chapter's actual code is executed
  under Python 3.13.5; different literal strings compare unequal, yielding False.
* CPython v3.13.5, Doc/library/random.rst, seed lines 72–98 and Notes on
  Reproducibility lines 461–476: integer seed initializes the PRNG; reusing
  a seed permits repeated sequences under the stated threading/version caveats.
  It is not a model score or a guarantee of cross-platform GPU determinism.
  The seed-42 split and choices sampler were actually independently replayed.
* PyTorch v2.9.0, docs/source/optim.md, How to use an optimizer / Constructing it /
  Taking an optimization step (lines 7–112): optimizers hold parameters and update
  them using computed gradients; parameters are learnable numeric tensors.
  This original version supports the general weight/update description only.
  It is not passed off as documentation for local torch 2.14.1+cpu; local API
  behavior is checked by actual execution and original project code inspection.
* Original project behavior.py at a864a60bbf72583afc9bbaf45e052bd4fe076c62:
  retrieved SHA 94ab92aa1b8017edbb8f5524ee598dce93756cf295377ac21515bf3c0a05bdc0,
  exactly matching safety.json's historical source registration. Inspected
  run_style lines 95–160, _safety_records 363–418, _safety_evaluations 421–451,
  run_safety 490–534. The current full-file hash differs, but each of these four
  complete function source segments is byte-for-byte unchanged. common.py,
  text.py, data.py, model.py, training.py also match the original run's registered
  full-file hashes. Thus current related code is mapped to the recorded run
  without claiming the complete behavior.py has the historical hash.
* Shannon, A Mathematical Theory of Communication (1948, corrected reprint of
  Bell System Technical Journal 27, 379–423, 623–656), original author paper
  retrieved from Harvard's mathematical text mirror. Read Introduction page 1,
  Section 6 pages 10–12 (entropy, zero uncertainty and conditional entropy),
  Section 9 page 15 (singular/invertible finite-state transducers and entropy
  loss). Personally rendered/viewed page 12 and confirmed H(y) >= H_x(y) and
  H(x,y) = H(x) + H_x(y). Used only for own transparent counterexamples: same
  red observation can leave count uncertain; a score can collapse distinct
  failure traces; a system must cover distinct task targets. This is not a claim
  that Shannon investigated model evaluations or recommends a modern ML metric.
* PyTorch v2.9.0, original torch/nn/functional.py cross_entropy lines 3375–3466:
  read complete function/docs, class-index target shapes, -100 ignore_index and
  excluded input gradients. Supports the general non-ignored-target supervision
  distinction. Actual target counts and current local execution are from
  torch 2.14.1+cpu, explicitly separated from the original v2.9 reference.

The recorded model uses TinyLM width 64, layers 2, vocabulary 264, heads 1,
max_length 128, learned absolute positions, manual attention, LayerNorm, GELU,
untied output, no experts. The style base trains on 49 arithmetic records for
1,000 updates. Two independent deepcopy branches then use SFT, AdamW lr .003,
batch 16 with replacement, auxiliary coefficient .01, gradient clipping 1.0,
900 updates, seed 42; prompt positions and padding have ignored labels -100,
assistant UTF-8 bytes and EOS are supervised. Generation is greedy, max 128 new
tokens for behavior/rewrites and 32 for arithmetic, also bounded by model
max_length 128. Exact matching compares raw answer IDs before EOS, not decoded
text alone. There is no optimizer step during any evaluation. Validation/test
are disjoint from parameter fitting and each other at full-prompt/family level;
major templates and document colors remain shared.
