# Reviewer-owned mathematical checks for 1.7

For three finite real logits z_i, define w_i=exp(z_i) and
p_i=w_i/sum_j(w_j). Each w_i>0; hence 0<p_i<1 for these three
candidates, and sum_i(p_i)=1 in real arithmetic. exp is strictly increasing,
so the ordering of logits is the ordering of the probabilities. The raw
vector [1,2,-1] has sum 2 and includes a negative component, so it is not
a probability vector.

For a common finite real c:

    exp(z_i+c) / sum_j exp(z_j+c)
    = exp(c) exp(z_i) / (exp(c) sum_j exp(z_j))
    = exp(z_i) / sum_j exp(z_j).

Choosing c=-max(z) makes all exponent arguments <=0 and the largest
exponential exactly 1. This prevents overflow from large positive exponent
arguments for finite representable inputs. This is not a promise about
NaN/infinite inputs, loss of input precision when adding enormous constants,
or absence of underflow of extremely small probabilities.

For [1,2,-1], subtracting 2 produces [-1,0,-3], whose mathematical weights
are [0.36787944117144233,1,0.049787068367863944]. The unshifted weights are
[2.718281828459045,7.38905609893065,0.36787944117144233], with denominator
10.475217368561138. Dividing produces
[0.2594964603424191,0.7053845126982412,0.03511902695933972].

If only the first logit increases from 1 to 3, the denominator increases
from exp(1)+exp(2)+exp(-1) to exp(3)+exp(2)+exp(-1). The second numerator
exp(2) stays fixed, so its normalized probability decreases. The new vector
is [0.7213991842739687,0.26538792877224193,0.013212886953789416].

For [B,V] logits, each fixed b defines a separate categorical distribution
p[b,v]=exp(z[b,v])/sum_j exp(z[b,j]). V is the last axis (dim=1 or -1),
whereas dim=0 instead normalizes each candidate across B distinct examples.

These are mathematical properties of the definition, supported by official
PyTorch formula/API sources and a bounded CPU execution. They establish no
model training, correctness of the highest-ranked candidate, or learned
prediction quality. Floating point sums are approximate: the original
float32 fence produced 1.0000001192092896, a difference of
1.1920928955078125e-7 from mathematical 1.
