# Independent policy and denominator derivation

These are possible architecture definitions, not measurements of a trained model.

For T tokens and top-k routing, there are A=T*k selected expert assignments.
For E experts, a proposed common per-expert capacity is
C=ceil(capacity_factor*A/E). The Switch v3 Equation 3 is the k=1 case;
the multiplication by k and integer rounding here are the lesson's explicit
assignment-budget convention, not additions quoted from Equation 3.

For counts n_i, excess assignments are o_i=max(n_i-C,0), and the lesson's
overflow rate is sum_i(o_i)/A. At k>1 this need not equal the fraction of unique
tokens with at least one missing expert contribution. There is no automatic
transfer of o_i to a different expert: a counts vector determines no new
selection or weight. Total capacity E*C >= A does not imply every n_i <= C.
In the lesson, (T,k,E)=(6,1,2), n=[5,1]:

| factor | C | E*C | excess | excess/A |
| --- | --- | --- | --- | --- |
| 2/3 | 2 | 4 | [3,0] | 3/6=0.5 |
| 1 | 3 | 6 | [2,0] | 2/6=1/3 |
| 2 | 6 | 12 | [0,0] | 0/6=0 |

The factor-only changes do not change the chosen indices. In the extra k=2
check, all six tokens select both experts, A=12, C=4, counts=[6,6],
excess=[2,2], and rate=4/12 rather than 4/6.

For an illustrative top-1 residual layer, let m_t be 1 if its selected expert
assignment is admitted and 0 on overflow. Let g_t be its selected gate weight,
f_e the selected expert, and f_default a specified output-shape-compatible
default function. Three possible policies require different contracts:

* Drop the expert contribution: y_t=x_t+m_t*g_t*f_e(x_t).
  When m_t=0 the residual x_t is still present. This does not define which
  token positions contribute to a training objective.
* Fall back to a defined default function on overflow:
  y_t=x_t+m_t*g_t*f_e(x_t)+(1-m_t)*f_default(x_t).
  Its weights, output scaling, compute budget, and training objective must be
  specified; this algebra claims neither improved accuracy nor a deployed policy.
* Reroute overflow to a chosen alternative e'_t: define e'_t, test the alternative
  capacity, define its gate weight, and combine its output. An alternative may
  also be full. Switch v3 Appendix B and Figure 11 explicitly discuss choosing
  the second highest expert and possible repeated overflow.

Thus the text lists design possibilities. The paper directly establishes fixed
capacity, residual-preserving skipped expert computation, and a rerouting example.
The generic fallback is a proposed contract as shown above, not attributed to
Switch as a tested quality improvement. The lesson promises no result for these
policies. Zeroing an expert contribution alone leaves the residual path, objective,
and any remaining top-k weight normalization as independent choices.

The configured local MoEFFN performs all selected assignments, so the arithmetic
overflow rates in this lesson describe a hypothetical bound, not observed drops.
A policy that skips computation changes the work executed; speed comparisons
must identify that change and report model quality separately.
