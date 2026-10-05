# Independent calculation for 7.8

The example has B=2 batch rows, T=4 physical sequence columns and V=5 candidate IDs. The reshaped contiguous artificial scores obey `L[b,t,v] = 20*b + 5*t + v` for 0<=b<2, 0<=t<4, 0<=v<5. The forty labels are dimensionless artificial numbers, not probabilities, training measurements or log probabilities.

For a nonempty Boolean row, replace each invalid physical column by -1 and take the maximum across T. This gives `max({t:valid[b,t]})`; `amax` returns a value, but here the values were deliberately the physical column indices. No gradient, optimization, token drawing or scoring evaluation occurs.

Row 0 true columns are {0,1}, so last=1 and scores are `20*0+5*1+{0,1,2,3,4}` = {5,6,7,8,9}. Row 1 true columns are {1,2,3}, so last=3 and scores are `20*1+5*3+{0,1,2,3,4}` = {35,36,37,38,39}. Paired row indices [0,1] and last indices [1,3] produce one entry per batch row for each untouched candidate column, hence [B,V]=[2,5]. They do not form a Cartesian product.

Counts minus one give [2-1,3-1]=[1,2]. This formula equals the final physical index only for a nonempty right padded row whose true columns form a prefix {0,...,n-1}. A left padded row or a row containing holes can have the same count and a different maximum. The exercise changes row 0 true columns to {2,3}; count remains 2, last becomes 3 and scores become {15,16,17,18,19}.

The synthetic integer score labels 0..39 have exact float32 representations and the index calculations are int64, so these comparisons require exact equality. There is no denominator, averaged metric or rounding in these score/axis examples. Additional model comparisons use an absolute tolerance of 1e-6 with rtol=0, comparing one newly initialized random TinyLM against itself for each input. They check index/mask/position mechanics and do not establish learned language quality.

All False leaves [-1,-1,-1,-1], whose maximum is -1. Advanced indexing accepts -1 as the physical final column; `gather(1,...)` rejects -1. Neither behavior supplies a valid input position. The independent guard explicitly rejects all-False and zero-column rows before reducing or selecting. These are verification helpers; the course fence demonstrates the mechanics and explicitly warns that a formal entrance must reject an empty input. The repository `generate` function explicitly supports unpadded input, so this section does not certify a padded batched generation implementation.

For rows with holes, selecting the maximum true physical column still works. Meaningful next-token scores additionally require that the valid mask and model position IDs used in the forward pass describe the intended ordered content. The independent TinyLM check removes PAD keys via valid, numbers true content consecutively via cumulative counts, and compares selected output with unpadded content. Correct last-index retrieval alone cannot repair a different forward computation.
