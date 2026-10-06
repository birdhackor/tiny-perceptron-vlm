# Independent inspection record

Reviewer task: /root/p6_fact_1_7. This is a new factual reviewer identity,
separate from the author and first reader. Date: 2026-10-06 UTC.

Read the current factual-reviewer instructions, complete section_facts.py
helper, and check_technical_reviews.py schema. Ran the checker's --help.
No old report or author/reader/review summary was read. The existing 1.7
report was copied as opaque bytes to history and hashed without JSON parsing
or displaying its contents.

Actual manuscript reading: 01.md lines 180-240 (tail of 1.5 and full 1.6);
original-byte extracted 1.7, lines 239-273; line-number inspection 239-283
(also first lines of 1.8); first-steps.md W.4, lines 111-134. Necessary
prerequisite sections 1.6 and W.4 have independent raw-byte snapshots.
01.md frozen whole-file SHA is explicitly the initial input, not a claim
that unrelated chapter sections can never change. No chapter introduction
is needed because this is not the chapter's first section.

Section 1.7 contains one Python fence and no image references, SVG references,
or inline diagrams. Accordingly no figure rendering/viewing was applicable.

Executed the original fence bytes unchanged in probe.py. Inspected its AST
calls and all result pointers in probe-results.json: /environment, /original,
/reference_float64, /float32_example, /stability_float32_plus_100, /batch_axes,
/exercise_cat_3, /wrong_dog_highest, /allclose_and_assert, and /scope.
All additional probes are short deterministic CPU operations on small tensors.
No training, GPU work, held-out evaluation, or model download was executed.
The installed environment is Python 3.13.5, torch 2.14.1+cpu, CPU/float32,
with torch git commit 5c4886908584029761b579af026dcfb627c84070.

Original official source inspections:

- PyTorch functional.py at installed commit: softmax lines 2176-2216,
  including formula, range/sum, dim argument and delegation to input.softmax.
- PyTorch _torch_docs.py at installed commit: torch.allclose lines 834-866;
  torch.exp lines 4310-4333. Defaults, tolerance formula and e^x were read.
- PyTorch SoftMaxKernel.cpp at installed commit: _vec_softmax_lastdim,
  lines 61-93, max reduction, subtract-max exp, sum, reciprocal and rescaling.
- Python 3.13 Language Reference, section 7.3, 'The assert statement', HTML
  lines 478-506 and the independently extracted plain-text section. This
  documents a false assertion raising AssertionError and the -O limitation.

The installed functional.py and _torch_docs.py files were independently
copied and their full-file SHA-256 hashes match the fetched official files
at the installed torch commit exactly. This is a source-version check, not
a claim that the fetched C++ file was rebuilt during this audit.

Initial docs.pytorch.org HTML requests returned HTTP 403. The failure is
preserved in fetch-manifest.json; no page from those URLs was inspected.
Official raw GitHub sources worked with normal TLS verification. v2.9.0
sources were initially fetched; only the kernel's locations were searched
before the installed commit was obtained. The final report cites the
installed commit, not an assumed match to v2.9.0. All official fetch URLs,
access timestamps, byte counts and full hashes are retained in the manifest.

Mathematical invariance, axis semantics, rounding and finite precision limits
were independently derived in derivation.md. In particular 'sum 1' is
supported as the mathematical/rounded description. The actual item() output
is retained in full; exact floating point equality to 1 is not claimed.

No manuscript, figures, or root progress file was edited by this reviewer.
