# 2.3 independent calculation (2026-10-05)

Input x is one batch row and three dimensionless features: [1,2,4].
Weight axis 0 indexes the two output recipes; axis 1 indexes the three input features.
W=[[1,0,2],[0,-1,1]], b=[1,0].

y_0 = 1*1 + 2*0 + 4*2 + 1 = 10.
y_1 = 1*0 + 2*(-1) + 4*1 + 0 = 2.
Matrix shapes are x[1,3] @ W.T[3,2] + b[2] -> y[1,2].
No averaging, sample denominator, physical unit, probability, or rounding is involved.

The exercise changes W[0,2] from 2 to 3. Delta y_0 = x_2*(3-2) = 4;
delta y_1 = 0 because output recipe 1 is unchanged. Hence y'=[14,2].

For arbitrary leading axes, y[b,p,j]=sum_i x[b,p,i]*W[j,i]+b[j].
There is no summation over the batch b or position p. [2,4,3] becomes [2,4,2].
To mix C positions each of width D, arrange each sample as C*D inputs first.
Repository ContextMLP uses embedding(contexts).flatten(1), preserving batch axis 0,
and Linear(context*width,width). This agrees with the lesson's requirement.

For the original non-square W, x[1,3] @ W[2,3] is invalid: inner lengths 3 and 2 differ.
A square counterexample: x=[1,2], W=[[1,3],[2,4]].
x@W.T = [1*1+2*3,1*2+2*4] = [7,10].
x@W   = [1*1+2*2,1*3+2*4] = [5,11].
Both shapes are valid; they compute different recipes.

Separate trainability probe only (not the original lesson having trained):
With zero targets and mean squared error over two scalar outputs,
L=(10^2+2^2)/2=52. dL/dy=[10,2].
dL/dW=[[10,20,40],[2,4,8]], dL/db=[10,2].
SGD at learning rate 0.01 gives W'=[[0.9,-0.2,1.6],[-0.02,-1.04,0.92]],
b'=[0.9,-0.02]. This confirms both registered parameters can be updated from loss.
It says nothing about learned prediction quality or generalization.

Tolerance: integer-valued original, exercise, rank-3 and transpose examples are exact
in float32; independently calculated outputs and actual tensors are exactly equal.
The original allclose has rtol=1e-5 and atol=1e-8; actual max absolute error is zero.
The independent SGD update uses rtol=0, atol=1e-6 for float32 rounding.

No SVG or other image is referenced by current 2.3; no figure was rendered or viewed.
This is absence of a figure in the reviewed source, not a rendering-tool failure.
