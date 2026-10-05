# 7.5 independent inspection

Reviewer: `/root/phase4_factual_coordinator/factual_7_5`. Date: 2026-10-05.
The assigned 7.5 section was read in full, directly from its current UTF-8 bytes.
The chapter introduction was also seen while reading the chapter file. I read 7.1–7.4 for tokenizer/label alignment and 7.6–7.7 for denominator and attention-mask context.
I did not read previous technical or reader reports, author conclusions, or other reviewers' judgments.
7.5 is not the chapter's first section, so an introductory summary is not an applicable review requirement.

## Original authority inspection

All network resources listed in `sources/download-provenance.json` were obtained directly from HTTPS authority hosts.
The saved HTML/Python/PDF bytes, rather than search results or source-library summaries, were inspected.
The original PDF front page identifies *Attention Is All You Need*, its original authors, NIPS 2017, and arXiv:1706.03762v7, 2 Aug 2023.
The official PyTorch documentation URL fixes version 2.9; the installed CPU environment is 2.14.1+cpu, commit `5c4886908584029761b579af026dcfb627c84070`.
This version difference is explicit. The API contracts involved here were additionally checked in original PyTorch `_tensor.py` and `nn/functional.py` at the exact installed commit.

- Transformer paper PDF pp. 3–4, §3.1 Decoder and §3.2.1 Eq. (1): decoder self-attention prevents reading subsequent positions, while permitted keys/values contribute to a weighted sum. Supports visibility/data-dependence; does not specify this repository's dialogue labels or guarantee a nonzero gradient for every parameter state.
- PyTorch 2.9 CrossEntropyLoss, class-index equations and `ignore_index` parameter (derived text lines 123–168, 223–226, 252–266): ignored targets have zero direct derivative with respect to the corresponding input logits and are excluded from the unweighted effective denominator. Relevant targets are integer class indices, no class weights or smoothing. Does not mean the upstream token embedding is removed or frozen.
- Original PyTorch installed-commit functional.py lines 3478–3571: cross_entropy arguments, ignore-index contract, candidate-class/input shapes, reduction and dispatch into the native loss. Lines 6366–6409 and 6480–6508 independently confirm query/key axes and the True-means-allowed mask convention. The sample uses the repository's manual attention, not SDPA; these semantics corroborate, rather than replace, inspection of manual_attention.
- PyTorch Autograd mechanics §How autograd encodes the history (derived text 132–148) and §Setting requires_grad (325–359): reverse differentiation traces operations using the chain rule; gradients accumulate into participating leaves; freezing parameters requires a distinct requires_grad operation.
- PyTorch Tensor.is_leaf (123–135), Tensor.retain_grad (123–126), Tensor.backward (123–138): non-leaf `.grad` is not retained by default; retain_grad keeps it; backward computes/accumulates derivatives, not an optimizer step. Original installed-commit `_tensor.py` 566–630 verifies backward semantics.
- PyTorch Embedding (123–176): integer lookup indexes `(num_embeddings, embedding_dim)` learnable weight rows. No padding_idx is set by TinyLM, and Q is an ordinary token ID. A whole-table non-None gradient does not establish which rows have nonzero derivatives.
- PyTorch Tensor.detach (123–140): returned tensor is detached from the current graph, requires no gradient, and shares storage. This cuts the path into the input embedding; it does not erase the forward numerical input.
- PyTorch Tensor.norm (123–126), torch.norm (123–220), vector_norm (123–183): default Frobenius norm equals the Euclidean p=2 norm for these one-dimensional real vectors, with no dimension selection. Installed-commit `_tensor.py` 888–902 dispatches Tensor.norm to torch.norm. The current docs mark torch.norm deprecated, but the operation is available and its checked one-dimensional behavior matches the lesson.
- PyTorch Tensor.item (123–130): extracts the scalar as a Python number; not a gradient operation. PyTorch manual_seed (123–135): selects the random generator seed. These complete the grouped API check of the original fence along with tensor indexing, backward and retain_grad.

## Repository contract and independent calculations

Read `data.py` 10–21 and 54–68: byte IDs are byte+8, Q is 81+8=89, Z is 90+8=98, A is 65+8=73; targets are shifted once.
For Q/A, X=[1,3,89,2,4,73] and Y=[-100,-100,-100,-100,73,2]. The scalar effective denominator is 2 answer tokens, not 6 input positions.
Read `model.py` 14–28, 31–86, 92–108: default one layer, one head, manual attention, untied input/output tables; embedding weights `(264,8)`; logits `(1,6,264)`; loss_sum reshapes to `(6,264)`, uses sum CE and ignores -100, rejects zero effective count; masked_loss divides by count.
Read `attention.py` 10–28 and 48–74: `[batch,head,query,key]` allowed mask, key<=query causal direction, valid masks keys, and a weighted sum of values. Read `modern.py` 42–61: the selected feed-forward path is positionwise, so the single attention layer is the only cross-position path for this example.

For finite unweighted class-index logits z, let M_t=1[Y_t!=-100], N=sum M_t=2.
L=(1/N) sum_t M_t (logsumexp_c z_tc - z_t,Y_t).
Thus dL/dz_tc=M_t/N*(softmax_c(z_t)-1[c=Y_t]); ignored rows are exactly zero.
For E_89, the upstream derivative is sum over supervised positions of (dL/dz_t)*(dz_t/dE_89).
It can be nonzero through attention despite dL/dz_2=0. Nonzero gradients are not guaranteed for every parameter state, and a tied output table could supply another path; this example explicitly uses the default untied table.
The two norms have different coordinate spaces (264 candidate-logit derivatives versus 8 embedding-coordinate derivatives). A raw numerical norm ratio is not a common causal-effect unit or an accuracy measure.

## Actual bounded executions

Original command: `.venv/bin/python docs/review-tools/section_facts.py course/chapters/07.md#7.5 --execute --output /tmp/phase4-7_5-original --timeout 45`.
It exited 0. Raw fence stdout: first-position logit gradient norm 0.0, Q-row embedding gradient norm 0.00571822514757514.
`original/` preserves the original section/fence/bootstrap bytes, helper execution command, stdout/stderr, environment/imported-module hashes and source hashes. Its temporary workspace/symlinks are not declared as required evidence.

`bounded_controls.py` is independently written and its entire actual stdout is saved:
- Original Q and exercise Z keep all ignored-logit rows exactly zero; Z row norm=0.004994203336536884, and absent Q row norm=0.
- Removing Q as an allowed key in the one-layer model makes Q-row norm exactly zero; output parameters still receive gradients.
- Detaching lookup embeddings preserves every forward logit exactly, but leaves embedding.weight.grad=None while output gradients remain nonzero.
- Omitting retain_grad leaves logits.grad=None, generates the expected PyTorch warning, and still computes Q embedding gradients.
- Query 4 reads key 2, and cannot read future key 5. Changing Q->Z changes first-answer logits by max 0.042634427547454834; changing only future A->B changes them by exactly 0.
- In float64, hand-derived CE=5.618060945615395, observed=5.618060945615394, absolute difference 8.88e-16. Analytic gradient max error=2.60e-18. Replacing ignored logits with large arbitrary values preserves loss exactly. All-ignored labels raise the repository's explicit ValueError.
- Independent sqrt(sum squares) differs from float32 Tensor.norm by 1.26e-10, within the predeclared 1e-8 tolerance.
- Every parameter remains exactly equal before/after backward. Optimizer steps=0, training runs=0. No model/data downloads, no pre-existing weights evaluation, no measured accuracy claim.

## Visual inspection and applicability

7.5 has no image reference (`svg_references=[]`), so its figure_consistency field is not applicable.
The already read preceding alignment diagram from 7.4 was additionally copied, rendered with Inkscape 1.4 to `context-figure/alignment.png`, and actually viewed.
The viewed diagram distinctly labels physical positions 0–5, X IDs and Y targets, gray/blue ignored rows and green effective rows. Q appears at X position 2 with ID89 and -100 target; assistant position 4 predicts A73; position 5 predicts EOS2.
Those labels agree with the example's raw X/Y. This is contextual evidence and does not turn an unreferenced figure into a required 7.5 figure.
Rendering emitted two library-initialization warnings, but exited 0 and produced readable Chinese glyphs and complete labels. A full HTML desktop/mobile page was not tested; no such test is claimed.
The code and explanations answer the needed data-flow question in 7.5: direct score gradient and upstream embedding gradient are explicitly separate; an additional diagram is not required for technical correctness.

## Judgment

All substantive claims verified within their stated scope. No substantive correction or unknown claim remains. No textbook or figure edits were made by this reviewer.
The example demonstrates a differentiable dependency, not learned answering ability or universal positive gradient guarantees. The wording '通常' and the explicit absence of a step preserve those limitations.
