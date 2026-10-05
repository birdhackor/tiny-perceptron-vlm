# 19.10 reviewer arithmetic and scope

For `Linear(8, 4)`, weight has axes `(output, input) = (4, 8)`.
The uncompressed layer has `4*8 + 4 = 36` FP32 values, hence `36*4 = 144` bytes.
The example's input has shape `(1, 8)` and output has shape `(1, 4)`.

The repository uses symmetric integers `[-7, 7]` at 4 bits and `[-127, 127]`
at 8 bits. Its per-channel reduction is `amax(-1, keepdim=True)`, so each
output row has one scale and the scale tensor has shape `(4, 1)`.
There are four FP32 scales (16 bytes), and the four unchanged FP32 biases
also occupy 16 bytes. Integer packing stores two 4-bit codes in one byte:
32 codes occupy 16 bytes. Thus 4-bit tensor storage is `16+16+16=48` bytes.
At 8 bits each of 32 int8 codes occupies one byte, giving `32+16+16=64` bytes.
These counts omit Python attributes, object overhead and serialized-container metadata.

For symmetric reconstruction, `weight_hat[j,i] = integer[j,i]*scale[j]`.
The scale is a weight-value interval per integer step; `2*0.5=1.0` exactly.
The linear output difference for the all-ones input is
`delta[j] = sum_i (weight[j,i] - weight_hat[j,i])`, because the identical bias
cancels. Absolute output error is bounded by the sum of eight absolute weight
errors for that output row. The fence takes the maximum across its four output
coordinates. It is an absolute activation-value difference, with no accuracy
denominator, percentage unit, or test-set aggregation.

The fixed CPU run observed max errors 0.046346306800842285 (4 bits) and
0.0037038326263427734 (8 bits). These observations concern one input and the
repository CPU implementation; they do not assert monotonic errors for every
input or preservation of an assistant's task performance.

The distillation calculation is
`0.5*CE + 0.5*T^2*mean_valid_tokens(sum_vocab p_T*(log p_T - log q_T))`,
with `T=2`. The raw KL term sums over the common vocabulary and then divides
by the number of valid answer positions. Teacher probabilities are detached.
The helper already multiplies by T squared. Hinton et al.'s soft-target CE
and forward KL have the same student gradient for a fixed teacher because
`KL(p||q)=CE(p,q)-H(p)` and `H(p)` is independent of student parameters.

Stored PTQ/student results were verified by counting their original records and
recomputing their documented exact-action/EOS and final-answer criteria. No
model was trained, downloaded, rescored or retained in a new weight file.
