# Independent factual inspection of 14.6

Reviewer task: `/root/phase4_factual_coordinator/factual_14_6`.
Date: 2026-10-05 UTC. Fresh single-section factual reviewer; no author or earlier
reader/technical report was read. No child agent was spawned. No chapter, figure,
training configuration or weights were edited. No training, GPU execution,
complete model evaluation, dataset download or model download was run.

The entire raw section 14.6 was read, including its source/results details.
Necessary context actually read: 14.5, 2.1, 4.6 and T.8. No chapter introduction
was reviewed. The whole chapter was copied as a **frozen input at this review's
start**, not read or certified in full; its actual snapshot and SHA are in
`inputs/chapter14-frozen-input.md` and `manifest.json`. Section and prerequisite
snapshots preserve raw bytes. Section 14.6 contains no image reference.

The input E has shape [V,D]; a Linear output table has [out,in]=[V,D]. Thus h
with last axis D produces V logits through h @ E.T. With the three rows given,
the individual dot products are 2,1,3. Sharing leaves 6 independent scalars
instead of 12; changing E[0,0] by 1 adds h[0]=2 to score 0. Changing E[1,1]
by 1 instead adds h[1]=1 to score 1. Integer/represented float values here are
exact. Original fence and exercise were executed on CPU, not inferred from
matching printed text. The original fence performs manual changes under
no_grad and no backward/optimizer update. The supplementary derivative check
does backward only, without an optimizer, to check a shared gradient buffer.

Press and Wolf's actual arXiv v3 PDF was downloaded from
https://arxiv.org/pdf/1608.05859v3. The first page identifies its title/authors,
arXiv identifier and date (21 Feb 2017). I read PDF pages 1-3 (extracted text
lines 1-200): section 1 describes U in R^(C x H), V of equal size, scores V h2
before softmax; section 3 sets U=V=S and explicitly sums updates from both
roles. That supports the mechanism and constrained common matrix. It does not
assert that every weight-tied architecture/initialization improves quality.

PyTorch source was read from the official pytorch/pytorch immutable git
revision 5c4886908584029761b579af026dcfb627c84070, exactly the git version
reported by the installed 2.14.1+cpu wheel. Read locations and concise primary
excerpts are saved in `sources/inspection-locators.json` and `*.inspected.txt`.
Embedding's weight shape, index lookup and normal initialization; Linear's
transpose rule, bias=False and shape; Parameter registration/default grad;
Module assignment and duplicate-removing parameter iterator; no_grad's
reverse-mode recording behavior; and load_state_dict's copy/assign contract
were inspected directly. `torch.no_grad` has factory and forward-AD exceptions;
neither is used by the original manual mutation code.

Serialization source notes at that same revision explain storage/view
preservation and state dict loading. A tiny BytesIO-only check preserved shared
storage in the serialized tensors, while loading into two independent modules
left distinct Parameter objects with equal values. Loading into an explicitly
tied destination retained one Parameter. This supports checking conversion
relationships, not a claim that normal saving always duplicates tied weights.
No weight file was written. Quantization's lower-bitwidth definition was read
from official PyTorch **v2.5.1** docs/source/quantization.rst lines 12-19;
it is an older conceptual reference, not an installed quantizer test. The
installed-revision and v2.9 quantization pages inspected here only redirect
development to torchao; they are not used to support the definition.
The first serialization.rst fetch returned HTTP 404; the real file at the same
revision is serialization.md. acquisition.json retains that failed fetch.

Existing empirical evidence was inspected only by named raw/provenance JSON
pointers after listing necessary object keys and types. Full source JSON is
preserved unmodified in raw/modern.json; selected values and actual inspected
pointers are saved separately. No notes/review/correction summaries or earlier
verdicts were read. The release/public-release JSONs were inspected for
top-level keys/types only and contributed no values/evidence. Candidate
locator manifests were used only to look for immutable primary objects.

The raw result revision is 4ef6555710e9915179a4a69738cdfabe80c299cf, CUDA
PyTorch 2.14.1+cu126, NVIDIA L4, Python 3.13.3, seed 42. Current model.py and
common.py match recorded code SHA; architecture.py differs. `git show` at the
recorded revision recovered architecture-measured-revision.py whose SHA
1ad0b0789318236d5e1a8fc1117cb65758bbd6e3ad62dc2d646a111ae1ea37a3 exactly
matches the report. It was AST-indexed before necessary node reads.
Read original nodes: _copy_matching 51-63, _clone_config 66-69,
_text_dataset 72-82, _nll 139-149, relevant _train slices 168-284,
_heldout 287-288 and run_modern 291-328. Method comparison and single-seed,
fixed-token-budget limitations were ordinary method descriptions, not result
answers. Current architecture node locations read before recovering that
original are _description 49-50, _copy_matching 53-65, _clone_config 68-71,
_text_dataset 74-84, _nll 141-151, _train 154-286, _heldout 289-290,
run_modern 293-330 and _parameter_budget 333-348. No results interpretation
constants or author correction summary was exposed.

Other repository method reads: model.py ModelConfig 15-28, TinyLM 54-89,
loss_sum/masked_loss 92-105; common.py seed/new_lm 41-47, split_records and
records_sha256 50-74, text_examples 77-97, _nll 106-119 and evaluate_lm
243-304. modern.py was AST-indexed only. build_course.py was AST-indexed;
build 175-236 was read to establish it is a whole-book notebook generator and
was not run. section_facts.py and checker schema were read as requested.

The recorded tied configuration has 124672 scalars versus baseline 141568,
exactly 264*64=16896 saved. CPU structural counting independently agrees.
Original _copy_matching is executed by selecting only that AST function. It
copies embedding/common matching tables, deliberately skips output.weight
for tied, and preserves output.weight is embedding.weight. Thus output values
differ from baseline's independent output initial table despite unchanged
common initial tables; a fixed hidden-vector probe confirms different scores.
This is initialization control verification, not reproducing initial training
NLL or establishing a cause for final heldout quality.

Raw NLL arithmetic is nll_sum / effective_tokens (natural-log cross entropy,
nats per scored target); not an unweighted mean over records/chunks. The
initial denominator is 329581 targets in 2789 chunks from 409 records. Training
uses 452102 token presentations, 16 chunks per sampled batch and 240 successful
updates with 0 skips in both groups. Validation uses 39256 targets, 334 chunks,
51 records; test uses 41914 targets, 352 chunks, 52 records. CPU recomputation
of all six selected NLL quotients agrees within 1e-12; displayed five-decimal
rounding is exact. A higher local NLL in this single-seed, fixed short-budget,
different-output-initialization comparison supports only the local observed
ordering, not universal harm from tying or a causal attribution to its
structural sharing constraint. Section 14.6 states that limit accurately.

Visual inspection: I viewed desktop-section.png, mobile-section.png and
prior-2.1-figure.png after actual Chromium 151.0.7922.173 rendering. The section
screenshots use reviewer CSS and frozen Markdown, not the published course
site. All relevant numeric prose is visible; mobile code needs horizontal
scroll in this isolated template. Markdown inside HTML details remains literal
in this minimal renderer, so these images are not evidence of website link
rendering. The necessary context SVG shows the 5-row table, lookup arrows and
shared dog-score recipe coherently. Its raw frozen SVG is retained; this is
context inspection, not a separate factual verdict on 2.1. Own section needs
no visual material beyond its explicit row vectors and dot-product example.
The first local-file navigation failed with ERR_BLOCKED_BY_ADMINISTRATOR;
render-first-failure.txt records the actual command/exit/exception. A changed
second attempt used page.set_content with identical raw bytes and succeeded.

Conclusion: no substantive unresolved claim or requested correction in 14.6.
No general weight-tying quality claim, trained-model reproduction, quantizer
acceptance, or actual published-site validation is included in this verdict.
