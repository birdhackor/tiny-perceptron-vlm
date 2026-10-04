# 13.14 independent hand calculation

Reviewer: /root/integration_technical_coordinator/fact_v2_13_14
Read date: 2026-10-04. These are hand derivations, separate from actual CPU execution.

For the one scalar prediction v and one observed reward r=1, the mean denominator is 1:
L(v)=(v-r)^2, dL/dv=2(v-r).
At v=0.4: L=(-0.6)^2=0.36 and derivative=-1.2.
Gradient descent v_next=v-eta*dL/dv increases v for positive eta.
At v=1.4: L=(0.4)^2=0.16 and derivative=+0.8.
Gradient descent decreases v for positive eta.
The direction is toward this observed target. Reaching it without overshoot depends on eta;
the lesson states the direction, not an unconditional convergence guarantee.

For random reward R at a fixed context s, E[(v-R)^2|s]
=(v-E[R|s])^2+Var(R|s). Thus the population squared-error minimizer is
v=E[R|s]. One sampled reward supplies a noisy regression target, not an estimate
of the entire mean by itself. Finite network approximation, finite sampled cards,
and the changing policy can prevent an accurate critic. This is consistent with
PPO v2 section 5's learned state-value function and squared-error target.

Softmax(log(p_i))=exp(log(p_i))/sum_j exp(log(p_j))=p_i/sum_j p_j=p_i
for a normalized positive finite categorical distribution. The lesson's p=(0.8,0.2)
and q=(0.5,0.5) each sum to 1; all entries are positive. Zero probabilities
would require careful -infinity handling and do not occur in this example.

KL(p||q)=sum_i p_i*ln(p_i/q_i), over all two available cards:
0.8*ln(1.6)=0.3760029033965885;
0.2*ln(0.4)=-0.1832581463748310;
total=0.1927447570217575, rounds to 0.1927 at four decimals.
If p=q=(0.5,0.5), both ratios are 1 and both logarithms are zero, so KL=0.
No stochastic action estimate, token denominator, padding mask, or sample-selection
denominator is involved; exact_kl returns one sum for each context.

The term -beta*KL in DPO v3 section 3 equation 3 has the same policy-to-fixed-reference
direction. PPO v2 equation 8 instead uses KL(old||new), a different per-iteration
trust penalty; it must not be substituted for this reference constraint.
In the repository's finite experiment, expected normalized RM reward and the
analytical current-policy reference KL are optimized as separate terms.
Its critic regresses only normalized RM scores. The OpenAI original train_policy
implementation (cbfd210) adds sampled token log-policy/reference penalties to rewards
before GAE and critic returns; this is a different explicitly documented recipe.

The PPO branch has four network roles: current policy (148 parameters), fixed RM
(241), critic (97), fixed reference (148). Old log probabilities are a frozen
per-rollout tensor record, not a fifth network. The separate DPO comparison branch
creates another policy instance; the lesson's four-network statement is scoped to PPO.
