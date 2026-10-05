# 5.17 independent derivation

Read source: course/chapters/05.md#5.17, original UTF-8 section SHA-256
73a38cebaa7b50e8c06417f91af572de9ff949613e857a2d049e44f34ea8a3bf.

For Linear(2, 1), weight W has shape (out_features, in_features) = (1, 2),
bias b has shape (1), input x has shape (batch, in_features) = (1, 2),
and output y = x W^T + b has shape (1, 1). Thus
y[0,0] = W[0,0] x[0,0] + W[0,1] x[0,1] + b[0].
At input [1,1], this is w1 + w2 + b, as in the manuscript.

L = output.sum() sums every element. Here there is exactly one element,
so L = y[0,0], with no mean, denominator, averaging, or physical units.
dL/dW = [[1,1]], dL/db = [1], and dL/dx = [[w1,w2]].
Setting the leaf parameters requires_grad=False excludes their new gradient
accumulation; it does not change the mathematical derivative with respect to x.
With fixed W = [[2,-3]] and b = [0.5], y = -0.5 exactly, and dL/dx = [[2,-3]].
These values are binary-exact float32 values. The 24-case CPU matrix checks
bitwise equality, the relevant leaf gradients and unchanged parameter values.

The independently executed exercise uses its own random initialization.
Its actual run prints W and x.grad as [[0.6714,-0.6219]] (four-decimal display);
the earlier helper run prints [[0.6496,-0.1549]]. Neither value is a claimed
fixed expected answer. Both original exercises execute their own allclose
assertion against the exact tensors from that invocation. torch.allclose uses
|input_i-other_i| <= atol + rtol*|other_i|, default atol=1e-8, rtol=1e-5.
The independent deterministic matrix additionally checks exact tensor equality.

For a trainable generator G = I_2, h = x G^T and the frozen reference
y = h [[2,-3]]^T + 0.5. The chain rule yields dL/dG = [[2,2],[-3,-3]].
Reference leaf gradients remain None. Wrapping the reference forward in
no_grad or inference_mode makes its new output have no recorded connection,
so backward cannot propagate to G through that output. This is tested only
on a small differentiable Linear reference, not arbitrary nondifferentiable
models or model training quality.
