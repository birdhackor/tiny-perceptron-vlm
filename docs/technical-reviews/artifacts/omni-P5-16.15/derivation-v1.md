# 16.15 reviewer derivation

This is an independent check of the stated artificial example and dependency
conditions, not a learned model result or timing experiment.

For a frozen deterministic encoder E with state bytes and all computation
settings fixed, each question/candidate branch has output g(E(state), q, c).
The recomputed and shared paths supply exactly the same M = E(state) to g.
If g does not mutate M or retain cross-question branch state, substituting the
same M leaves its mathematical output unchanged. Floating-point equality must
still be checked under the intended implementation. This does not establish
that E or g makes correct decisions.

For three question invocations, the recomputed path evaluates E three times;
the shared path evaluates E once. Both still evaluate each question/candidate
query, query-key products, weighted values, and scores. This operation-count
fact does not prove elapsed-time savings, especially where copying or cache
management adds work.

The example uses two material rows and dimension d = 2. Its encoder is
M = X + softmax(X X^T / sqrt(2)) X. The two input score rows are
[1/sqrt(2), 0] and [0, 4/sqrt(2)]. Softmax gives first-row weights approximately
[0.6697615493, 0.3302384507] and second-row weights
[0.0558072192, 0.9441927808]. Therefore M is approximately
[[1.6697615493, 0.6604769013], [0.0558072192, 3.8883855616]].
The scalar Python reference recomputes all branch weights, hidden features,
scores, and probabilities independently of torch matrix multiplication; its
full results are in inline-execution-v1.json.

The artificial recurrence is r_new = 0.5*r_old + x + 0.1*x_previous.
Starting with r=0 and previous=0: x=1 gives r=1; x=2 gives r=2.6 and previous=2.
From this same prefix, x=3 gives 1.3+3+0.2=4.5; x=4 gives 1.3+4+0.2=5.5;
restarting x=3 gives 4.5 again. Reusing branch endings instead gives
0.5*4.5+4+0.3=6.55, then 0.5*6.55+3+0.4=6.675.
Deleting newly appended KV pairs leaves the separate r and convolution history
unchanged; the output artifact checks this explicitly.

The old CandidateScorer joins state, question, candidate into each token row.
Block supplies a valid-key padding mask to SDPA and does not set is_causal.
The official SDPA default is is_causal=False. Material queries can therefore
read question/candidate keys. Cropping those material positions is generally
not equivalent to encoding state alone. The lesson's bidirectional artificial
counterexample illustrates this dependency; no formal C network was rerun.

For the fixed Qwen configuration, the 32-entry text_config.layer_types list
contains the repeated pattern [linear_attention, linear_attention,
linear_attention, full_attention] eight times: 3*8=24 linear layers and 8 full
layers. The pinned Kev card identifies those linear layers as Gated DeltaNet.
The exact Kev prefix method copies its layer objects and mutable state
dictionaries, then reorder_cache uses index_select to allocate branch tensors.
The CPU cache fixture tests this method under a deliberately mutating fake
backbone and exposes contamination when the two state-dictionary copies are
removed. It does not run the actual Qwen network or validate its quality.
