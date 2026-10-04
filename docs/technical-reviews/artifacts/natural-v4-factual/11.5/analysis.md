# Independent factual analysis of 11.5

Reviewer: `/root/v4_review_coordinator/factual_v4_11_5`, fresh factual role.
The assigned whole raw section hash is `dd3f827b602ad35b1dcd4e96409c7871af942d65d6c42f473ee89a640101ebd7`.
No introduction is owned. Current raw 11.2, 4.5, 7.5, 11.3 and T.6 were read;
none contains an SVG. No necessary figure exists to render or view.
The previous assigned report was copied byte-for-byte without inspecting its content.

## Derivation and predictions before interpreting results

With dense GELU FFN, one attention head, untied input/output tables, learned
position embeddings and default LayerNorm, a width-w block has two LayerNorms
(4w parameters), four bias-free attention matrices (4w²), and an FFN with
two matrices (8w²) plus biases (5w). Total: B(w) = 12w² + 9w.

The image projector maps visual width 16 to language width w and has
P(w) = 16w + w = 17w parameters. Language storage is
L(w,n) = 2·264w + 128w + n·B(w) + 2w.
Vision storage is 48·16 + 16 + 16·16 + 2·16 + 16·16 + 16 = 1,344.
Audio encoder storage is 16·16 + 16 + 2·16 + 16·16 + 16 = 576.
The wrapper total is L(w,n) + 1,344 + 576 + 2·P(w).

Therefore w=8, n=1 gives B=840, P=136, L=6,104 and wrapper=8,296;
partial gives 136+840=976. Adding a second block gives all=9,136 while
projector=136 and partial=976. Four blocks give all=10,816; partial remains 976.
w=64, n=2 gives B=49,728, P=1,088, L=141,568, partial=50,816 and all=145,664.
CPU module enumeration agrees exactly. These are parameter element counts,
not training examples, FLOPs, throughput, or measured allocation peaks.

One FP32 parameter element occupies 32/8=4 bytes. For default AdamW, the
first and second moment buffers are parameter-shaped, plus a scalar step.
Gradient storage and other activations/workspaces also contribute to training
memory. AMSGrad adds another buffer; allocator/backend/dtype details affect
actual usage. No fixed multiplier establishes all live memory.

For the transparent supervision example, 3 valid positions/use × 10 uses = 30.
The CPU mask check uses class-index labels [-100,1,2,-100,3], counts three
valid positions and verifies zero direct logits gradients at ignored positions.
This does not remove ignored input tokens from later attention or guarantee
equal optimization difficulty for equally many targets.

## Primary sources actually inspected

* Vaswani et al., arXiv:1706.03762v7 (2 Aug 2023), PDF p.3 §3.1 and p.5 §3.3:
  stacks of attention/position-wise feed-forward layers. The original paper uses
  post-norm and encoder cross-attention; the repository's pre-norm causal block
  is checked in its own source and is not asserted to be the identical model.
* Liu et al., arXiv:2304.08485v2 (11 Dec 2023), PDF pp.4–5 §4.1–4.2,
  especially p.5 Stage 1/Stage 2: projection-only alignment followed by training
  projection and LLM while freezing the vision encoder. This establishes a real
  method for language adaptation, not superiority of the toy model's partial scope.
* Li and Hoiem, arXiv:1606.09282v3 (14 Feb 2017), PDF p.1 §1 and p.2 §2.1:
  fine-tuning shared parameters can change earlier-task performance; feature
  extraction and fine-tuning have distinct tradeoffs. Its experiments are CNN
  classification, not direct evidence of these VQA table values.
* Kingma and Ba, arXiv:1412.6980v9 (30 Jan 2017), PDF p.2 Algorithm 1 and §2:
  m_t=β1m_(t-1)+(1−β1)g_t and v_t=β2v_(t-1)+(1−β2)g_t² are stored histories.
* Official PyTorch source at commit
  `5c4886908584029761b579af026dcfb627c84070`, retrieved through raw GitHub:
  Module.requires_grad_, torch.numel documentation, Adam._init_group,
  AdamW.__init__ and CrossEntropyLoss class-index ignore_index formula.
  All five fetched whole files equal the installed package files byte-for-byte.
  The actual runtime reports torch 2.14.1+cpu, Python 3.13.5 and no CUDA.

Original retrieval hashes, URLs, versions and dates are in original-source-receipts.json.
Full external papers/source are confined to ignored research, not copied here.

## Actual execution and empirical scope

verify.py executes the section's selection loop; actual `_freeze`; the actual
current `scripts/train.py` freeze branch extracted by AST; one image-only
forward/backward/AdamW step; supervision-mask checks; and independent arithmetic
on the original VQA record. Image-only all mode leaves audio parameters with
grad=None, no optimizer state and exactly unchanged weights. This is one CPU
mechanism check, not a replication of the CUDA learning experiment.

The ordinary CLI's two-block partial mode opens both blocks and both projectors,
counting 1,952 at width8; fixed VQA partial opens only the last block and image
projector, counting 976. Input/output tables remain frozen in the CLI partial branch.

The existing original VQA report records CUDA, NVIDIA L4, Python 3.13.3,
torch 2.14.1+cu126, seed42, full schedules and step_scale=1. Training source
specifies AdamW, lr=0.003, four examples per update and 160 updates. All three
non-replay branches copy the same projector checkpoint; their recorded before
evaluations are identical, and their upstream training report exactly equals
the original projector report. Checkpoints are not present locally and were
not downloaded, loaded or independently trained. These origin facts are an
audit of original records and matching relevant program source.

Splits are 36 train queries derived from 18 images at offsets -2,-1,0;
12 validation queries from six images at offset1; and 12 test queries from
six images at offset2. Families do not overlap. Each holdout has six color
questions and six shape questions; questions about the same image are correlated.
Generation is greedy, at most 16 new IDs; matching uses raw byte IDs before EOS.
All 72 stored validation/test outputs were independently rescored from IDs.

Sampler recomputation reproduces every update's target count and total3,830,
including EOS. The nonempty replay list causes rng.random() to be consumed
before each image choice even when replay probability is zero. An initial
reviewer script omitted this draw, failed its total assertion and was preserved
as verify.initial.py / verify.initial.stderr.txt. Correcting that probe to the
actual source behavior makes all assertions pass; this was a probe issue,
not an inconsistency in the lesson or original records.

Validation/test scores are exactly projector3/3, partial10/12 and all12/9,
each out of12. Projector test color=0/6, shape=3/6; all color=6/6,
shape=3/6, with all three square questions answered circle. Partial validation
errors are row1 red→gre and row8 circle→square; test is12/12.

Current model.py, multimodal.py, modalities.py and common.py hashes equal the
recorded versions. The runner file's whole-file hash differs; both hashes and
the exact limits are retained. It does not affect the matching inspected
sampling/training/scoring source, but the original GPU run is not claimed to
have used the current runner bytes. No timing or GPU memory number is presented
as a new measurement. A single seed, six images/12 correlated queries per
holdout and this small synthetic world cannot establish a universal freeze winner.

No unresolved factual issue was found in the assigned current section.
