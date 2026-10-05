# Fresh independent review of 1.13

Reviewer: /root/phase4_factual_coordinator/factual_1_13. Date: 2026-10-05.
This file records my direct inspection, not a previous technical report's conclusions.
I read the current raw section completely, the reviewer instructions completely,
the checker schema completely (including a second read of the truncated check-status
branch), section_facts.py completely, and build_course.py BOOTSTRAP lines 26–61.
Additional current textbook context actually displayed: course/chapters/01.md
lines 355–471 (part of 1.10 and all 1.11–1.12), and course/chapters/16.md
lines 236–306 (16.6 plus the displayed beginning of 16.7). Only 1.13 is judged here.
I did not read a previous 1.13 report, previous technical report conclusions,
or another reviewer's inspection notes. Shared-source receipts were used only
to identify available immutable official URLs; my snapshots were freshly fetched.

## Exact source and applicability

The saved original/section.md is the raw UTF-8 section from its ## 1.13 header
up to the ## 1.14 header. It has no CRLF and SHA-256
c9ab11c75ec8b335272e56ee19aa9044b92f88659bdbed26f4c994cb886b90d5.
The source begins at course/chapters/01.md line 472; its single Python fence
starts at line 479. It imports torch and nn and calls built-in PyTorch APIs;
there is no project model/helper dependency in the fence.
There is no Markdown image, HTML image, or SVG reference in 1.13. The table is
checked against arithmetic and execution. No figure rendering or viewing was
performed or claimed for this section.

## Independent arithmetic, axes, units, and tolerance

The parameter and loss are scalars (shape []), so there is no batch axis or
reduction denominator in the original example. L(w)=w². For a nonzero h,
[L(w+h)-L(w)]/h = [(w+h)²-w²]/h = 2w+h. Its limit as h tends to zero is 2w.
At w=2, L=4 and each newly computed derivative is 4. The derivative's units
are loss units per parameter unit; the .grad accumulator has those same units.
There is no physical unit attached to this toy scalar. No parameter update is
performed, so w stays 2 throughout.

Initially grad=None. Backward supplies 4, the next fresh-forward backward adds
4 to produce 8, setting grad=None removes that accumulator, and a third
fresh-forward backward supplies 4 again. Expected printed tuple: (4,8,4,2).
If the clear line is commented out, three contributions give 4+4+4=12;
first=4 and second=8 still satisfy second==2*first. That assertion cannot
establish the third gradient's clearing. Values 2,4,8,12 are exactly
representable in float32, so the original and exercise comparisons use exact
equality and no rounding tolerance. Negative w=-1.5 gives -3 then -6.
Changing the second loss to 3w² gives contributions 4 and 12, sum 16: the
twice-first relationship is specific to equal losses at unchanged w.

For the extra optimizer check with lr=0.1, SGD changes w from 2 to
2-0.1*4=1.6; float32 gives 1.600000023841858. Absolute tolerance 1e-6 was
declared before checking. zero_grad changes the gradient buffer, and step
changes the parameter. step alone does not clear grad in this check.

The two-microbatch control has two one-sample scalar losses (w-0)² and
(w-1)², each divided by the common sample count 2. Their derivatives at w=2
are 2 and 1, sum 3. The full-batch mean derivative is also 3. The denominator
is two equal-weight samples, not a number of optimizer steps or language
tokens. This is a bounded arithmetic/API control, not evidence for arbitrary
model equivalence, memory savings, speed, GPU AMP, or training quality.

## Official source passages personally read

All eight snapshots were retrieved from the PyTorch project's own GitHub
repository at 5c4886908584029761b579af026dcfb627c84070, the exact git_version
reported by .venv torch 2.14.1+cpu. Thus there is no installed/latest version
substitution. source-receipts.json records HTTPS URLs, HTTP 200 responses,
access timestamps, byte counts and SHA-256. Actual source inspection:

- official-parameter.py lines 30–63: Parameter is a Tensor subclass considered
  a module parameter; requires_grad defaults True; __new__ passes it to
  _make_subclass for this tensor input.
- official-tensor.py lines 566–625: backward computes gradients of the current
  tensor with respect to leaves by the chain rule, accumulates into leaves,
  recommends zeroing or setting grad=None, and defaults graph retention to
  create_graph=False unless explicitly requested otherwise. The call delegates
  to torch.autograd.backward and contains no optimizer update.
- official-tensor-docs.py lines 1104–1120, 2786–2810, 4888–4905,
  6553–6572: clone and square link to their torch equivalents; item returns a
  standard number for a one-element tensor; grad is None initially and later
  backwards add gradients into it.
- official-torch-docs.py lines 2908–2935 and 11028–11057: clone returns a copy,
  while retaining differentiation relationships; square creates a new tensor
  containing elementwise squares. Cloning is not asserted to detach.
- official-autograd.md lines 1–62 and 194–215: forward operations construct
  the graph, each iteration recreates it, x² saves x for backward, and only
  leaf tensors requiring gradients accumulate into .grad. The section's
  cautious wording about releasing part of graph materials agrees with
  backward's retention contract and the actual saved-tensor error.
- official-optimizer.py lines 1048–1107: zero_grad resets optimized tensor
  gradients; set_to_none=True is default and assigns p.grad=None; False
  detaches/disables gradient tracking for p.grad then zeros it. None and zero
  may differ for parameters that did not receive gradients; this review does
  not generalize their equivalence beyond the exercised receiving parameter.
- official-sgd.py lines 79–145 and 336–380: step collects parameters with
  gradients and dispatches SGD; with ordinary lr, zero momentum and zero
  weight decay the update is param.add_(grad, alpha=-lr).
- official-amp-examples.md lines 126–188: explicitly documents adding
  gradients over microbatches, dividing loss by the accumulation count in
  that equal-batch example, and performing step/zero_grad at the effective
  batch boundary. Its AMP-specific scale rules were read but not executed;
  only the general intended accumulation and timing claim is used for 1.13.

## Execution scope

The unchanged fence ran first through section_facts.py in a separate guarded,
offline CPU process, exit 0, stdout '4.0 8.0 4.0 2.0', no stderr/guard events.
The original bytes, exact helper command, bootstrap, environment and output
were copied unchanged from /tmp into permanent original/ artifacts.
probe.py then executed the unchanged fence and its exact requested one-line
exercise, and independently checked saved-tensor reuse failure, retained graph,
clone vs alias, both optimizer clear variants, SGD's parameter update,
negative w, unequal loss contributions, and a two-sample accumulation control.
All its assertions passed on one-thread CPU float32, Python 3.13.5,
torch 2.14.1+cpu; no model/data download, GPU work or full training occurred.

No substantive discrepancy remains in the current 1.13. The later 16.6
measurements and model implementation are not re-certified by this review.
