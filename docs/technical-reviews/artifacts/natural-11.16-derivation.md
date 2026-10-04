Independent mathematical inspection by /root/natural_factual_11_16, 2026-10-04.

For reference prefix r[:i] and prediction prefix p[:j], let D(i,j) be the
minimum unit-cost insertion/deletion/substitution count. D(0,j)=j and D(i,0)=i.
Partition an optimal alignment by its final step: inserting p[j-1], deleting
r[i-1], or aligning r[i-1] with p[j-1]. These exhaust the possibilities, giving
D(i,j)=min(D(i,j-1)+1, D(i-1,j)+1,
           D(i-1,j-1)+(r[i-1] != p[j-1])).
Each candidate is also constructible from its shorter optimal prefix alignment,
so this is equality, not only a bound. Induction on i+j establishes correctness.
Repository previous[j] is D(i-1,j), current[-1] is D(i,j-1), and previous[j-1]
is D(i-1,j-1). Only the preceding row is needed; returning previous[-1] gives
D(len(reference),len(prediction)). Empty strings are covered by initialization.

This alignment derivation describes reference -> prediction. The chapter's
prediction -> reference convention is also correct: invert every insertion to
a deletion, every deletion to an insertion, and each substitution to its
inverse. Unit costs are preserved, so the two minimum counts are equal.
The repository returns only the total count, not directional error categories.
In the chapter's direction, omitted 北 must be INSERTED and an extra 北 must be
DELETED. CER still divides by the reference length in either direction.

今天去臺北 contains 今,天,去,臺,北: five code points, fifteen UTF-8 bytes.
今天去台北 differs at one position and can be repaired in one substitution;
unequal strings need at least one edit. Thus exact=False, edits=1, CER=1/5=0.2.
Omitting 北 produces edits=1 and the same denominator five, not four.
Applying the stated replacement 台 -> 臺 to both inputs makes them identical,
so the separately normalized comparison is exact=True, edits=0, CER=0.
The report retains the original comparison; the function itself never normalizes.

今日休館明日開放 and 明日開放今日休館 each have eight code points and the same
character counts (日 appears twice). The same character bag does not imply an
identical sequence. The two 日 positions agree; replacing zero-based positions
0,2,3,4,6,7 gives a six-edit path. The executed official TorchMetrics v1.8.2
full-table edit-distance function independently confirms minimum six, matching
the repository's two-row implementation. CER is 6/8=0.75. Including a newline
between the four-character lines makes both strings nine code points; the
minimum remains six and CER becomes 6/9. Removing newlines does not repair the
line permutation. These are constructed string examples, not measured OCR.

Python str positions are Unicode code points. One U+20000 character occupies
one position but four UTF-8 bytes. A combining sequence e + U+0301 consists of
two positions; a woman-technologist emoji joined with U+200D consists of three.
The algorithm does not count user-perceived grapheme clusters and performs no
automatic Unicode normalization. An empty reference has no CER denominator;
text_error_report deliberately returns None while preserving edits/exact.
Extra prediction characters can produce CER greater than one (e.g. reference
字 vs prediction 字字字 gives 2/1=2), and ordinary Levenshtein has no unit swap.

The independent executable enumerates 225 ordered pairs of binary strings of
length 0..3. Its oracle is breadth-first search over actual strings after each
individual edit, rather than the repository's DP recurrence. Every minimum
count and both directions match. The search alphabet is sufficient for these
binary examples: inserting/substituting a third character cannot help reach a
binary target faster than inserting/substituting the target character.
The deletion-then-insertion upper bound limits the number of explored edits;
a path shorter than that bound cannot grow beyond starting length + the bound.

No SVG is referenced in 11.16; figure review is not applicable. The necessary
linked 11.13 distinguishes its aligned equal-length toy calculation from
general CER, and linked 20.6 describes empty-reference/text-presence and line
ordering conventions. Neither linked text nor this string-only check supplies
evidence of natural-photo OCR quality, model training, or hallucination rates.
