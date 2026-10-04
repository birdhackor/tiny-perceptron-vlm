# Independent 9.6 derivations

Reviewer: /root/v4_review_coordinator/factual_v4_9_6, fresh factual/technical context.

Let s=[True,False,True,False], a be observed refusal. The refusal numerator is sum(a_i for s_i=True), denominator count(s=True)=2. The normal non-refusal numerator is sum(not a_i for s_i=False), denominator count(s=False)=2. No all-four average appears in either measure.

All-refuse a=[T,T,T,T]: selected refusals [T,T] -> (1+1)/2=1.0; normal negated [F,F] -> (0+0)/2=0.0.
Never-refuse a=[F,F,F,F]: (0+0)/2=0.0 and (1+1)/2=1.0.
Ideal a=[T,F,T,F]: (1+1)/2=1.0 and (1+1)/2=1.0.
First-false a=[F,F,T,F]: (0+1)/2=0.5 and (1+1)/2=1.0.
These binary fractions are exactly representable as float32. The actual CPU execution checks equality without rounding tolerance. Empty subgroup has denominator zero; torch.mean returns NaN, consistent with the undefined mean, so reporting 0 or 1 would invent a measured proportion.

The test partition's kinds are: denied permission 3, allowed permission 3, known count 3, missing count/clarification 3, true premise 1, false premise 1, color/document 3. Total=3+3+3+3+1+1+3=17. Normal=3+3+3+1+1+3=14. The independent generation-ID audit recomputes (appropriate-refusal /3, normal-exact /14, normal-refusal /14): before (0,0,0); safety-only (3,12,0); mixed (3,13,0). Total exact matches thus are 0,15,16; this does not combine arithmetic test scores into refusal scores.

Both 2/2 and 200/200 are arithmetically 1.0. For independent comparable Bernoulli trials with unknown success probability p, the probability of all successes is p^n; for any 0<p<1, p^200 < p^2. Thus more independent representative successes provide more evidence than two successes. This is an illustrative derivation, not a confidence claim about the highly correlated toy-template families or a guarantee of future safety.

W.3 prerequisite figure: row means (60+70+80)/3=70 and (80+90+100)/3=90. Column means (60+80)/2=70, (70+90)/2=80, (80+100)/2=90. dim=1 removes the three-exam axis, leaving two student means; dim=0 removes the two-student axis, leaving three exam means.
