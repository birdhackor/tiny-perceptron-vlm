# Independent factual review of 10.8, initial source

Reviewer: /root/v4_review_coordinator/factual_v4_10_8, fresh context.
Date: 2026-10-04. Section raw SHA-256:
`bc5dbcdfd673cf0cfd8846449f6cefff005d6abb40289d458ef0059652e46cac`.

I read every UTF-8 byte, including blank lines, of section 10.8 and its explicit prerequisites 10.7 and 4.6. See source-first.receipt.json for hashes and exact chapter line ranges. 11.3 and 11.7 are onward navigation about training/retention; 10.8 does not assert their measured results, so they are not factual dependencies here. No introduction is required because 10.8 is not a first section. Section 10.8 directly references no SVG. The prerequisite figures add nothing needed to verify these two software paths, so no figure was selected or claimed to have been viewed, and figure_sha256 is {}.

## Original authority inspected

I actually downloaded official PyTorch repository originals at the exact installed torch git commit `5c4886908584029761b579af026dcfb627c84070`. Runtime reports PyTorch 2.14.1+cpu. Complete fetched originals stay under ignored outputs/natural-v4/factual-research/10.8; retrieval URLs, SHA-256, statuses, versions and access date are in authority-retrieval*.json. Public evidence here contains only my analysis and receipts.

- torch/random.py, lines 49–86: manual_seed seeds device generators and the CPU default generator. 0 is a seed integer, not a model score.
- torch/nn/modules/module.py, lines 1976–2022: assigning a Module stores that same child module value in _modules. Lines 2894–2932: train(False) recursively sets training flags, eval delegates to it, and only affected module types change behavior. eval does not generally guarantee determinism or disable gradients. Current TinyLM/DenseFFN/manual attention contain no Dropout; the lesson's statement about some training randomness is an API-level motivation.
- torch/_torch_docs.py, equal API lines 4241–4263: equality tests sizes and elements, without an approximation tolerance; NaNs fail equality and dtype itself is not distinguished. The actual logits are same float32 dtype and finite. unsqueeze API lines 12475–12504 inserts a size-one dimension and shares data.
- torch/nn/modules/linear.py, complete Linear class lines 53–134: y=x A^T+b and output replaces final in_features axis by out_features. TinyLM uses bias=False and no post-head softmax.
- torch/nn/functional.py, complete cross_entropy definition lines 3478–3570: inputs are predicted unnormalized logits. Complete scaled_dot_product_attention documented API lines 6367–6540: its reference formula is masked softmax of scaled QK followed by multiplication by V. The API explicitly warns that backend choices can change floating-point output. The current project's manual and sdpa branches implement that same mathematical formula for the no-padding, single-head example.
- docs/source/notes/numerical_accuracy.md: read complete document. Lines 5–17 explicitly distinguish mathematical equality from bitwise floating-point equality; lines 23–38 also discuss batch/slice differences. This directly contradicts the section's general implication that mathematical equivalence is sufficient for torch.equal.
- docs/source/notes/randomness.md: read complete document. Lines 5–14 disclaim cross-release/platform/CPU-GPU reproducibility; lines 27–45 describe the same-environment fixed-seed conditions. No cross-platform reproduction is claimed here.

The first numerical_accuracy.rst request returned HTTP 404. Its genuine failure remains in authority-retrieval.json. I corrected the filename to the actual .md source and then downloaded and read it. No unverifiable authority is used as evidence.

## Independent computation and scope

The exact lesson code ran in the installed CPU runtime with no optimizer. Both [1,20,30] and the exercise [1,40,50] returned (1,3,264). Shape derivation: one batch × three input positions; width 8 is transformed by the 8→264 output Linear; 1×3×264=792 logits per sequence. Both entry comparisons returned True and max absolute difference 0. The registered language object and every language parameter object were identical. A forward hook observed one unchanged batch [1,20,30] and no altered keyword arguments. Default TinyLM positions are 0,1,2. No language state changed during forwards. Reinitializing TinyLM after manual_seed(0) reproduced its state and logits in this same runtime.

Inputs [1,5,4] and [1,6,4] without modalities each raised ValueError with the actual message `placeholder 缺少配對圖片／聲音`. Legal IDs are 0–263; 5 and 6 are special modality markers, while the lesson's tested ordinary sequence IDs contain neither.

The result establishes only the tested wrapper identity and code-path behavior; no train/test split, trained task accuracy, GPU training, broad retention benchmark or speedup was reproduced. Denominators, seed=0, zero training updates, stdout, stderr, exact commands, runtime and timing scope are saved in probe-* and probe-results.json. A controlled deepcopy weight change preserved every argmax but changed the scores; that is an algebraic/API demonstration, not a training experiment. It supports the section's distinction between score equality and task quality.

## Substantive correction required

Chapter 10 line 237 says that mathematical equivalence should imply all scores are identical, and the code uses torch.equal. For the actual wrapper, equality is justified by direct dispatch into the same language object with the same batch input and operation path, not by mathematical equivalence alone.

My bounded precision_probe.py reused seed=0, [1,20,30], width=8 and identical state_dicts while changing the implementation from the manual attention branch to sdpa. Both implement softmax(Q K^T / sqrt(D) + causal mask) V, but 369 of 792 output elements differed, max absolute difference 2.980232238769531e-07, while all argmax IDs stayed unchanged. A separate float32 regrouping check also gave (1e20 + -1e20)+3 = 3 but 1e20+(-1e20+3) = 0. Both are exact real-arithmetic value 3. This is evidence of the stated boundary, not a failure of the lesson's actual wrapper.

Suggested correction: replace the mathematical-equivalence implication with a statement that this no-modality wrapper directly invokes the same text model, preserving input batch shape and operation path, so exact equality is appropriate for this example. If discussing different implementations/backends, describe toleranced numerical comparison separately. No other substantive defect was found in section 10.8.

Initial verdict: revise, one unresolved precision-condition issue. The official checker will only check schema and current byte identities; it cannot establish factual truth.
