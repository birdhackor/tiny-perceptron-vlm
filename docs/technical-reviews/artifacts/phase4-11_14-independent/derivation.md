# 11.14 independent calculation

Input axes are RGB channel, row, column: `(3,32,32)`. Red occupies
rows `[8,16)`, columns `[0,4)`; blue occupies the same rows, columns `[4,8)`.
The original method reverses columns `[0,8)` and so interchanges these bars.
The nonzero region belongs wholly to summary grid row 1, column 0.

For 32 input samples and 4 output bins, the official adaptive-pooling
`start_index` and `end_index` rules give `[8j,8(j+1))`. Each summary cell
therefore sums 64 samples **per channel**, not 192 intermingled RGB samples.
Red and blue each have 32 ones in this cell. Its RGB mean is `(32/64,0,32/64)`
= `(0.5,0,0.5)` in either picture. Every other cell is zero. This also follows
from an independent reshape to `(3,4,8,4,8)` and reduction over within-cell
row and column axes 2 and 4.

At all 64 affected coordinates, two channels change: red 1→0 and blue 0→1,
or conversely. Thus there are 128 unequal scalar elements out of 3072 RGB
values, but 64 changed color coordinates out of 1024 coordinates. The green
channel never changes. `.flatten()` retains the channel-major, row-major,
column-major array order; its resulting 3072 values differ between pictures
and reshape exactly to the original input.

If the red 8×4 bar is moved to columns `[8,12)`, total RGB sums remain
`(32,0,32)`. Original grid cell `(1,0)` becomes `(0,0,0.5)` and new cell
`(1,1)` becomes `(0.5,0,0)`. Two summary scalar values change; global color
totals alone still do not locate the red bar. Regional averaging therefore
does not discard *all* positions: this counterexample depends on the swap
remaining within one averaging cell.

Two distinct inputs with the same pooled representation supply exactly the
same input to any deterministic function of that representation. Such a
function cannot assign opposite left/right answers from this representation
alone. Distinct ordered pixel arrays retain this difference, but this fact
does not establish that a trained model uses it or understands natural scenes.
Object-name sets are likewise compatible with several spatial arrangements
or actions. Position and relation evidence plus appropriate question/answer
supervision are needed for the corresponding task. Boxes and image feature
sequences are available representations; neither automatically guarantees
relation accuracy.

All inputs are exact zeros and ones. The means 0 and 0.5 are exactly
representable in binary floating point, so the verification uses exact equality.
This is a deterministic information-flow calculation: no model, training,
gradient update, benchmark score, existing checkpoint evaluation, or natural
image result was generated.
