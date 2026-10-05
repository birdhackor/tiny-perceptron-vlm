# Fresh original-source inspection, 5.17

Reviewer: /root/phase4_factual_coordinator/factual_5_17. Accessed 2026-10-05.
No old technical/reader report body or other reviewer judgment was read.
No children were spawned. Read only the current 5.17 manuscript, its two
original fences, tool bootstrap, the four requested methodology/schema files,
and newly fetched official original sources. This is not a chapter-opening
section; no introduction review applies. There are no figures or measured
training results in this section, and no model/data/training recipe was run.

All official sources were fetched directly from raw.githubusercontent.com/
pytorch/pytorch at 5c4886908584029761b579af026dcfb627c84070, the installed
torch.version.git_version. See sources/fetch-receipts.json for HTTPS URLs,
HTTP 200 statuses, UTC timestamps and byte hashes. Nine installed Python
source files match the corresponding official source bytes exactly. This
uses the installed revision, not an assumed latest-release documentation API.

I personally read these original source ranges, not search excerpts:

- docs/source/notes/autograd.md lines 12–34: reverse AD graph and grad_fn
  entry point. Lines 182–230: requires_grad, at least one required input,
  only eligible leaves accumulate gradients, parameter freezing. Lines
  234–320: default/no-grad/inference modes and inference tensor restriction.
  Lines 328–348: eval/train are orthogonal to no_grad/inference and affect
  module-specific training behavior.
- torch/nn/modules/module.py lines 2670–2697: parameters() iterates registered
  parameters recursively by default. Lines 2894–2955: train sets training,
  recurses to children, eval calls train(False), module.requires_grad_ loops
  over parameters. It does not globally change the grad context.
- torch/nn/modules/linear.py lines 53–134: y=xA^T+b, axis shapes, Parameter
  creation for weight/bias, random initialization and forward F.linear
  with no use of self.training. torch/nn/functional.py lines 2382–2406
  confirm the original F.linear API and shapes.
- torch/nn/parameter.py lines 30–62: Parameter registration, default
  requires_grad=True and no_grad factory exception.
- torch/nn/modules/dropout.py lines 35–73: training random zeros with
  probability p and scaling 1/(1-p), eval identity, forward passes
  self.training. torch/nn/functional.py lines 1473–1499: dropout forwards
  that bool to the primitive. No claim about regularization quality is made.
- torch/autograd/grad_mode.py lines 22–86: no_grad changes/restores the
  thread-local grad flag; factories and forward AD are documented exceptions.
  Lines 213–295: inference_mode extra restrictions, no automatic eval,
  disabling forward AD, no version tracking and context implementation.
- torch/_tensor.py lines 566–626: backward is chain-rule differentiation,
  accumulates gradients in leaves, scalar seed may be omitted, defaults
  select leaf tensors used in the graph. No optimizer update occurs here.
- torch/_torch_docs.py lines 835–867: allclose's elementwise inequality
  and tolerances. Lines 8837–8865: ones shape, ones values and default
  requires_grad=False. Lines 11267–11317: sum without dim sums all elements.
- torch/_tensor_docs.py lines 4126–4144: requires_grad_ mutates the tensor
  flag in place; 5028–5034: Tensor.sum aliases torch.sum; 6614–6640:
  requires_grad is not proof that .grad is populated, and grad_fn=None
  for user-created leaves while only eligible leaves accumulate .grad.

Executed facts:

- section_facts.py extracted raw bytes and executed both untouched fences
  under the installed CPU .venv, exit 0. Its bootstrap sets seed 42; the
  exercise's second layer is newly created and replaces the first.
- The second untouched fence was also invoked in its own fresh Python
  process with -I and no bootstrap or first-fence state, exit 0. This
  verifies that the exercise is independently runnable and the old other,
  no_grad block and assertion are not needed.
- 24 Linear cases = train/eval × default/no_grad/inference × trainable/frozen
  parameters × input gradient off/on. Correct output flag, grad_fn,
  leaf gradients, failure when no graph exists, and unchanged parameters.
- 6 Dropout cases show eval identity and training zeros/scaling regardless
  of grad context. The identity reuses the old input, so even in no_grad
  the returned pre-existing leaf can have requires_grad=True. The text's
  False/None statement is scoped to the newly computed Linear output;
  it does not claim every object returned within no_grad changes flags.
- 3 generator/reference cases check the upstream gradient path exactly.
  Factory/nested enable_grad, grad flag restoration, stale .grad preservation,
  inference tensor backward restrictions, and forward-AD differences are
  explicit bounded checks. inference_mode is an additional requested
  boundary probe, not an extra subject claimed to be taught by this section.
- contracts.stderr.txt contains a PyTorch forward-AD import FutureWarning
  about torch.jit.script deprecation. All assertions and processes succeeded;
  the warning is preserved and does not indicate a numerical/test failure.

Supported scope: this section's default reverse-mode AD on new differentiable
Linear forwards. Freezing prevents new parameter gradient accumulation; it
does not clear existing .grad or enforce immutable values against manual
writes/optimizers. Fresh layers in the original examples have no stale grads
and no optimizer. No global claim that no_grad disables forward AD or
changes train/eval state is inferred. No inference speed benchmark was run.
