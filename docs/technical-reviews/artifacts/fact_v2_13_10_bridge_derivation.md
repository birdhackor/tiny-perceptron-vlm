# Score-gap bridge recheck

Read the actual DPO arXiv:2305.18290v3 PDF, section3 p3 equations(1)-(2) again. Equation(1) models conditional comparison p(y1 preferred to y2 | x), not p(answer factually correct). For scores c and r under the same x, divide numerator and denominator of exp(c)/(exp(c)+exp(r)) by exp(c): p=1/(1+exp(r-c)). With d=c-r this is 1/(1+exp(-d)), the logistic sigmoid. Equation(2) uses the chosen winner's -ln p.

exp(x)=e^x; e=exp(1)=2.718281828459045, rounded to three decimals2.718. d=0 gives exp(0)=1 and p=1/(1+1)=1/2 exactly. d=2 gives exp(-2)=0.1353352832366127 and p=0.8807970779778823; their four-decimal displays are0.1353 and0.8808. Substituting the explicitly rounded intermediate0.1353 instead gives0.8808244516867788, still0.8808 to four decimals; the displayed approximation is not an exact equality.

For every finite real d<0, -d>0, exp(-d)>1, so 1+exp(-d)>2 and 0<p<1/2. For finite real d generally exp(-d)>0, hence0<p<1. Float32/64 may round to endpoints for extreme inputs; the mathematical mapping and this small example are the scope. No absolute correctness target or truth calibration follows from the comparison model. A labeled chosen winner can receive model probability below0.5 if its estimated reward is currently lower.

Current code block matches the previous personally executed code exactly. The short CPU recheck executes that whole code plus rejected=1 and common+10, and checks math.exp/log against both equation(1) and torch.float64 sigmoid/preference_loss. Tolerance for finite toy analytic values1e-15; literal four-decimal stdout comparison; no training. Existing165-context/825-pair and portable checkpoint evidence is retained because its source code, result and tensor payloads remain unchanged.
