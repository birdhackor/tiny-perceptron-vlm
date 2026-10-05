# 2.5 independent arithmetic and information scope

For fixed theta and no extra side channel, f_theta(suffix) is the same candidate
distribution for equal suffixes. An equally weighted two-label set at one suffix
has expected negative log likelihood -0.5 log p(circle)-0.5 log p(square),
minimized at p(circle)=p(square)=0.5 (if those are the only possible labels).
Changing parameters does not create a color-conditioned distinction for a pair
whose presented input is identical. Random samples can differ, but their
distribution cannot use a missing color cue. This is conditional on the specified
fixed-window feed-forward model, not a claim about external memory or metadata.

Five input positions of D coordinates concatenate to C*D coordinates in general.
A dense map with H outputs has H rows of C*D multipliers: H*C*D weights,
plus H biases. Doubling C doubles that matrix, not the whole model.

At V=17, D=H=16, the actual model has VD + HCD + H + VH + V = 577 + 256*C
parameters. C=1/3/5 therefore gives 833/1345/1857. C=2 gives 1089 total,
showing why only the input-to-hidden weight part doubles.

The two five-code-point prefixes differ only at position 1. Their suffixes at
C=1/3/4 coincide; C=5 differs. The exercise prefixes have eight code points,
differ only at position 1, coincide for C=7 and differ for C=8. These all use
ordinary CJK code points, not bytes or a general tokenizer count.

For the published table, the raw historical receipt reports seed42/width16 and
200 updates for each context. The exact same 9/1/2 documents yield 103/11/22
targets after adding one end-boundary target per document. Each default CE is
the mean of target NLLs (natural-log units), not a document macro mean or accuracy.
The original historical 26f34eb/CUDA-build-on-CPU receipt is read as an existing
result. An already existing 5d60e35/CPU-build run has separate raw receipt and
checkpoint hashes; this review only evaluates those already available weights.
Evaluation mean versus sum/denominator and explicit log_softmax-gather sum are
checked in probe-result.json with specified float32 tolerances. The five-decimal
table and its three-decimal chart labels round the original historical values.

The synthetic rule for the first example does not define the 12-document field
corpus, which enumerates four colors times three shapes. Its target is every next
character, including field boilerplate and a document boundary. The table does
not test whether a model learned the first example's red-circle/blue-square rule.
The original manuscript calls this another experiment and states the toy rule
is not a real-world color-to-shape law. With one validation document and changing
parameter counts, the comparison supports only this run's descriptive trend,
not a general optimum or an isolated causal effect of context length.
