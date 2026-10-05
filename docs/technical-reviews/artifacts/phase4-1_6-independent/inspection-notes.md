# 1.6 independent factual inspection, 2026-10-05

Reviewer task: `/root/phase4_factual_coordinator/factual_1_6`; fresh context.

I read the reviewer instructions and the complete checker schema before reviewing.
I did not read the old 1.6 report, report history, or their conclusions.

## Current manuscript read

- `course/chapters/01.md`: chapter introduction (lines 1–4), preceding 1.5 (lines 160–201), all of 1.6 (202–238), and the first 1.7 paragraphs (239–245) to distinguish scores from probabilities.
- The 1.6 byte snapshot is `original-run/section.md`, SHA-256 `9e3255326b0c910464b89c128a9fcc7a237025a78ff2eae967783d8106bd64ac`; the current bytes still matched at the final inspection.
- I read the complete `docs/review-tools/section_facts.py` contract and the `scripts/build_course.py` BOOTSTRAP before execution. The helper executes the extracted original fence with the repository bootstrap; it guards against network, subprocesses and writes outside its artifact directory. This run had no blocked guard events.
- I also inspected the chapter 2 introduction and its Embedding locations solely to check the final navigation sentence; this is not a review or approval of section 2.1.

My summary: this section changes from counts to a manually specified, trainable lookup table. IDs specify rows; the three returned values are candidate scores. It shows initialization and forward lookup, not optimization or measured learning.

## Original external materials actually inspected

All original sources were fetched directly from the official `pytorch/pytorch` repository with HTTPS and normal TLS verification. The sources used for the claims are pinned to runtime git commit `5c4886908584029761b579af026dcfb627c84070` (installed PyTorch `2.14.1+cpu`), accessed 2026-10-05. Byte comparisons confirmed the official Embedding, Parameter, functional and grad-mode Python files are identical to the installed files.

- `torch/nn/modules/sparse.py`: Embedding documentation lines 14–55 and 69–112; constructor, weight creation, initialization and forward lines 134–197. Its weight is a `(num_embeddings, embedding_dim)` Parameter; no default max_norm or padding index is active here. The API supports arbitrary index shapes; the one-character context limit comes from this lesson's design, not an intrinsic Embedding limitation.
- `torch/nn/parameter.py`: Parameter lines 30–57 and `__repr__` lines 84–86. It registers module parameters, defaults requires_grad to True, and prefixes its representation with `Parameter containing:`. Creation/default tracking says nothing about whether training occurred.
- `torch/nn/functional.py`: embedding lines 2509–2560 and 2599–2621, plus cross_entropy lines 3478–3538. The embedding path only renormalizes when max_norm is supplied; this default path selects stored rows. Cross entropy accepts predicted unnormalized logits, consistent with score values being signed and unconstrained by a probability sum. This inspection does not claim the lesson executes a cross-entropy loss.
- `aten/src/ATen/native/Embedding.cpp`: embedding_symint lines 37–54 and backward dispatch/CPU implementation lines 56–135. One-dimensional indices perform `weight.index_select(0, indices)`, establishing row selection and identical scores for identical final IDs. Backward constructs a separate gradient tensor; it does not perform an optimizer step.
- `torch/autograd/grad_mode.py`: no_grad lines 22–86. The context switches grad mode and restores it; no_grad does not permanently disable the existing Parameter's requires_grad flag.
- `docs/source/notes/autograd.md`: graph history lines 12–35, requires_grad lines 194–230, and no-grad mode lines 277–298. The doc separates forward graph construction, backward gradient accumulation, and parameter mutation/initialization under no-grad.
- `torch/_tensor_docs.py`: zero_ lines 6280–6288, requires_grad/is_leaf lines 6613–6636. zero_ fills self; requires_grad signals gradient computation, not completed training.
- `docs/source/tensors.md`: indexing example lines 54–64, requires_grad example lines 84–94, and mutation naming convention lines 112–118. An underscore suffix marks mutation of a tensor; assignment alters the selected cell.

An initial set of v2.9.0 source snapshots was acquired, but the conclusions use the exact runtime-commit snapshots above. The runtime-commit `.rst` autograd URL returned HTTP 404 because the document is now `.md`; acquisition of the `.md` original succeeded. This was a source-path change, not an unavailable conceptual source or failed software check.

## Actual CPU verification and support limits

The original Python fence ran with the unchanged source bytes in `.venv` through section_facts, exit 0. All code, command, environment, stdout and stderr are retained under `original-run/`. The independent bounded probe then re-executed the same original bytes, checked the exact values and shapes, and ran the requested one-cell exercise. Its source, modified fence, result, environment and stdout are retained here.

- Original scores: weight `[[0,2,0],[0,0,0],[0,0,0]]`; output for `[0,2]`: `[[0,2,0],[0,0,0]]`, shape `[2,3]`. The column ID ordering means `2 > 0 = 0`, so 看 ranks first and 貓/狗 tie. All shown integers are exactly representable in float32; exact equality was appropriate.
- Exercise: adding `[2,0]=3` produced `[[0,2,0],[3,0,0]]`; cat output and the unrelated middle row were unchanged.
- zero_ returned the same object with the same storage address and zero values. A no-grad computed result had requires_grad False and no grad_fn, while the Parameter retained requires_grad True and remained a leaf. The same computation outside the context tracked gradients. Removing no_grad from leaf zero_ produced the expected in-place leaf RuntimeError.
- Forward lookup preserved all parameter bytes and its version counter. Original code never called backward, used an answer or called an optimizer. One additional sum.backward probe populated `[[1,1,1],[0,0,0],[1,1,1]]` in `.grad` and left the weights unchanged. This is a small autograd boundary check, not model training or proof of learning quality.
- A separate signed-score table returned a negative value and a row sum of 5 unchanged. It confirms the storage/lookup operation imposes no probability normalization.
- Parameter count derivation is V rows times V candidate columns = V²; for V=3 it is 9, and a four-character table has 16 cells. For the text's `顏色=` / `形狀=` thought experiment I expanded the toy vocabulary to include `=` before looking it up. Both final IDs were 3 and both returned `[12,13,14,15]`. The original three-character table cannot encode `=`; the manuscript example describes the general one-character score-table design, not an executable input to its three-character fence. No claim is made that Embedding itself is incapable of receiving several indices.

## Figure and applicability inspection

This section contains no Markdown image reference, SVG, HTML image/link-to-image or visual diagram. Its only table-shaped content is numerical prose and printed tensors. Figure consistency is therefore not applicable; no rendering tool was invoked and no visual inspection is claimed. This does not waive any numeric or software check.

No unresolved substantive inaccuracies were found. There are no empirical performance claims, GPU runs, data/model downloads, uploads, optimizer steps, training loops or manuscript/figure edits in this review.
