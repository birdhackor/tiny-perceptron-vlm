# 11.2 actual full-source recheck, round 2

Reviewer: /root/v4_review_coordinator/factual_v4_11_2, original independent factual owner.
Current entire raw section was personally reread, including blank lines, at SHA cd808d019acb4250d8e5e8377ba4868261d6bcacf458e5aefe52c6b0687786ac. All six necessary linked prerequisites W.6, 10.6, 5.17, W.2, 5.4 and 5.5 were personally read in full and saved with current SHA receipts. This is not the first numbered chapter section. There is no necessary SVG in the current section or any of these six prerequisites.

The first genuine revise remains unchanged: report-first.json SHA 47218a4cfce0fb7e8102a758c708f8737ff0a0f003cd3d693c3807c80eb79eed; source-original.md SHA 67aa88c3b5b026fefb59c3572125bfbc397b0d166b0b5e371a3a83674de01cf0; original probe.py/stdout/stderr/result and ignored original downloaded authorities are preserved. This round does not replace their hashes or present first-round evidence as a new execution.

## Complete current claim recheck and reuse boundary

The new Python fence is byte-identical to the first fence (SHA 07ed6499a1897512554e51ab3fc9a2c42768f322fcafd2256016c4ddee3d1ffc). Full registered project files and relevant installed official PyTorch source files retain their inspected hashes. I rechecked all current prose claims: name/parameter unpacking, flags and lists, shared parameter identity, optimizer grouping, initialization without backward/step, parameter sizes, exercise block naming/count, differentiable frozen-language route, no_grad, eval, and membership changes. Thus the first actual CPU execution is honestly reused for unchanged main/example code and those unchanged mechanisms. It still means 136/976, not a new training result. Counts independently remain projector 8*16+8=136 and final default block 2*(8+8)+4*(8*8)+(32*8+32)+(8*32+8)=840, hence 976.

## Original-authority reinspection for the changed facts

The original fixed-commit files previously downloaded were reread at these exact locations, their full SHA receipts checked, and installed-source byte identity reconfirmed. They are official v2.14.1 at installed commit 5c4886908584029761b579af026dcfb627c84070; raw file URLs have the form https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/<path>.

- docs/source/notes/autograd.md lines 194–235, Setting requires_grad: only leaf tensors with requires_grad=True accumulate new backward gradients; unchanged recorded-chain/no_grad/eval conditions were rechecked against the already read note.
- torch/nn/modules/module.py lines 2894–2955, train/eval/requires_grad_: the flag-changing loop does not clear .grad or rewrite an optimizer. No new claim about an absolute no-update barrier remains in the opening.
- torch/optim/adamw.py lines 19–48: AdamW inherits Adam and enables decoupled weight decay. Default lambda=.01 is positive in this snippet.
- torch/optim/adam.py lines 139–158, _init_group: only group parameters whose .grad is not None are selected. Lines 395–457 increment the selected parameter step, apply p *= (1-lr*weight_decay), and update the first moment from current gradient and history. Lines 458–551 update second moment, perform bias correction and subtract the normalized history-based step. A None gradient skips this whole path, including state evolution and decay; zero remains eligible.
- torch/optim/optimizer.py lines 1048–1109, zero_grad: set_to_none=True removes the stored gradient; set_to_none=False zeroes an existing tensor and therefore is not interchangeable. Constructor/add_param_group references previously inspected explicit group storage and insertion.

## Independent predictions and actual small new CPU execution

Command: .venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/11.2/round2/revision-probe.py
Python3.13.5, PyTorch2.14.1+cpu, one CPU thread. Deterministic scalar probes use float64 and no random input; exact prerequisite snippets use their original float32 defaults. No dataset, split, benchmark timing, GPU or long training was involved. Each independent optimizer branch uses at most two updates.

Flag/new-versus-retained prediction: p=1 computes grad2 using ordinary backward, then p.requires_grad_(False). A second differentiable p*u computation with u=3 preserves that old p.grad=2 without adding a new gradient to p, while u receives grad1. AdamW([p],lr=.001) may consume the retained gradient; the actual value becomes 0.998990000005. Then zero_grad(set_to_none=True) plus another step leaves the parameter and its step/m/v history exactly unchanged. Excluding a frozen tensor from the optimizer leaves it at1 even when its old grad2 remains. These tests directly verify the two corrective safeguards stated in the current text.

History-only prediction (g1=1,g2=0,beta1=.9,beta2=.999,eta=.1,eps=1e-8): after first update m=.1,v=.001. Second zero-gradient update has m=.09,v=.000999; mhat=.09/(1-.9**2), vhat=.000999/(1-.999**2); normalized step = .1*mhat/(sqrt(vhat)+eps) = .0670058244658113, which is nonzero despite current g2=0. With decay0, actual first value1.900000001 becomes1.8329941765341886, exactly matching the independent formula within the configured1e-12 tolerance. With decay.2, the additional multiplicative factor is .98: actual1.860000001 becomes1.7557941765141885. In the parallel None-gradient branches neither value nor optimizer history changes; the step remains1. The probes deliberately freeze the flag before these second calls and establish that the stored gradient, not the flag, determines eligibility.

Necessary new prerequisites were also checked as actual code: 5.4 gives first-step m=[.1,10],v=[.001,10],bias-corrected [1,100] and [1,10000], with updates approximately[.001,.001]; changing100 to-100 changes only the corresponding update sign. 5.5 gives2*(1-.1*.2)=1.96 (actual float32 value1.9600000381), while no-decay/zero-gradient and decay/None-gradient exercises both leave2. These directly support the new links and prevent interpreting learning rate as a fixed coordinate delta.

## c8 disposition and conditions

The current opening now says to decide which parameters receive *new* gradients, then inspect optimizer membership and actual changes. The next paragraph explicitly says switching the flag does not clear .grad and that AdamW can use retained gradients. The phase paragraph explicitly advises setting old gradients to None and excluding frozen parameters; it also explains that zero gradients can still permit history/decay updates and links the now-read 5.4/5.5. The original counterexample is now described by the text rather than contradicting it. c8 is genuinely resolved after rereading and new source/CPU checks.

The new shrink-toward-zero description concerns the decay component when the selected parameter has a gradient tensor and the configured decay is positive with0<eta*lambda<1, as holds for this lesson's defaults; it does not claim that the full AdamW displacement always heads toward zero. None skips the selected-parameter update; zero does not. The section's wording says parameters may still change, so it does not promise every zero-gradient case changes, including zero history/zero decay. New-gradient permission and membership remain separate from actual movement and learned task quality. No unresolved substantive issue remains in the fully reread current section.
