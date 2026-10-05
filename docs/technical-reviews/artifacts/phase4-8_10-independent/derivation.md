# Independent arithmetic and rule check: 8.10

The section defines a binary rule, not a model probability: arithmetic is correct AND the analogy maps two groups of two to the combined total. The four answers are handwritten.

- Index 0: two pairs of chopsticks imply 2 + 2 sticks, total 4; the stated total combines the two pairs. The rule permits 1.
- Index 1: the total 5 is false since 2 + 2 = 4; the star imagery does not supply the required grouping/combining. Label 0.
- Index 2: two plates, each two apples, combined four apples, explicitly maps all requested relations. Label 1.
- Index 3: repetition of 4 gives the correct numeral but no analogy or grouping. Label 0.

Thus H=[1,0,1,0], J=[1,1,1,0], E_i=1[H_i=J_i]=[1,0,1,1]. A=(sum_i E_i)/N=3/4=0.75 (dimensionless; N=4 supplied answer-label pairs). The disagreement set is {i:H_i != J_i}={1}; index 1 means the second answer under zero-based indexing. Human positives total 2; agreement counts total 3, including agreement that index 3 is bad. The two totals are different quantities, even though judge positives also happen to total 3 in this example.

Changing only J_1 from 1 to 0 yields E=[1,1,1,1], A=4/4=1, and no disagreement. All-zero supplied judge labels produce A=2/4=0.5 while judge positives equal zero, an additional discriminating counterexample.

All listed fractions have finite binary representations and the execution is expected to match exactly (no tolerance or rounding needed). The calculation measures agreement with the supplied human labels; it does not prove the human labels always true, identify a causal bias, or estimate general performance. No confidence calibration or judge parameter update is performed.

The appended exercise materials in variants/variant-results.json both have 62 Unicode code points by independent construction. The useful explanation explicitly supplies two groups and a merge; the repetitive explanation does not. Their labels [1,0] are manually evaluated under this section-specific rule; the script only stores/prints those labels and checks code-point lengths. These are not generated judge outputs or measured verbosity-bias evidence.
