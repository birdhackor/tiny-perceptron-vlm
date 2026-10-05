# Independent arithmetic and scope derivation, 15.5

For expert rows A=(1,0), B=(0,2), weights (.7,.2):
- Selected total s=.9. Direct sum .7A+.2B=(.7,.4).
- Normalized weights=(7/9,2/9), output=(7/9,4/9) ≈ (.7778,.4444).
- Swap rows only: (7/9)B+(2/9)A=(2/9,14/9) ≈ (.2222,1.5556).
- Swap both rows and weights: commutativity of the two addends preserves output.
- Sum contracts the expert axis of shape (2,2) against weights shape (2,), producing (2,). Concatenating rows produces (4,).
- Nonnegative normalized weights sum to 1, so output is on the line segment between expert vectors. There is no constraint on proximity to original x: for x=(100,100), the same expert vectors produce (7/9,4/9), far from x. This is a logical counterexample, not an experiment on a trained model.
- With a single positive selected probability p, p/p=1 and d(p/p)/dp=(p-p)/p²=0, holding the selection fixed. Softmax probabilities in the stated mathematical setup are positive. This derivation does not claim that finite precision underflow or changing a discrete selection is differentiable.
- FFN expert combination Σ_i g_i E_i(x) is separate from outer residual x+h. The original Transformer paper §3.1 and §3.3 explicitly distinguish residual and FFN computations; current model.Block lines 43–50 do the same.

Numerical tolerances: float64 probe absolute tolerance 1e-14; controlled float32 actual MoEFFN forward tolerance 1e-7. The original fence returns float32 Python values with binary tails after .round(decimals=4); the book's decimal values are correct at the stated four-decimal precision, not exact printed string representations. No backward or optimizer step occurs in the original fence. The single-p normalization derivative is separately checked in a bounded CPU graph; no training is performed.

Existing update count check: all seven named original variants each request 180 attempts, record 180 completed attempts and 180 successful optimizer updates, zero skipped updates, and 337,761 effective prediction-target tokens. This describes the saved run, not a new run and not a claim of performance superiority. The original revision's training method increments updates after finite gradients and scaler.step(optimizer), and its full source SHA matches /code_sha256/scripts~1course_experiments~1architecture.py. Installed modern.py and model.py also match recorded code hashes. No weights, dataset download, GPU execution or full evaluation were used.
