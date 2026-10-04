# 11.2 independent factual review, first round

Reviewer: /root/v4_review_coordinator/factual_v4_11_2 (fresh).
Read the complete raw UTF-8 section, including trailing blank lines, starting at chapter line 44; SHA-256 67aa88c3b5b026fefb59c3572125bfbc397b0d166b0b5e371a3a83674de01cf0.
Read W.6, 10.6, 5.17 and W.2 in full; exact originals and SHA receipts are adjacent. This is the second numbered section, so a first-section chapter introduction receipt is not applicable. 11.3 is a navigation pointer to the subsequent gradient/result lesson, not a prerequisite for this parameter-roster demonstration. Neither this section nor its four necessary prerequisites contains an SVG; no figure rendering is needed.

## Independent arithmetic before the CPU check

The actual projector is Linear(16,8). Its weight is 8 by 16 = 128 scalar elements; its bias is one scalar per output, hence 8. Sum = 136. The exercise enables the last of ModelConfig.layers=1 blocks, so its name prefix is language.blocks.0.
That width-8 block has two LayerNorm layers, each with weight 8 and bias 8: 2*(8+8)=32. Q/K/V/out are four bias-free 8 by 8 matrices: 4*64=256. DenseFFN hidden width defaults to 4*8=32. Its up projection has 32*8+32=288, and down projection has 8*32+8=264, for 552. Block total = 32+256+552 = 840. Combined with projector: 840+136=976. These are exact integer counts, not an empirical model-quality result.

A fixed differentiable linear operation y=2*x1-3*x2+1 on x=[4,5] produces -6 and gradient with respect to x=[2,-3], even when both weight and bias are frozen. This transparent hand derivation was also checked in CPU code.

## Actual short CPU work

Command: .venv/bin/python docs/technical-reviews/artifacts/natural-v4-factual/11.2/probe.py
Actual Python 3.13.5, PyTorch 2.14.1+cpu, official commit 5c4886908584029761b579af026dcfb627c84070, float32 CPU, one thread, seed 20261004. The official v2.14.1 tag was fetched afresh and resolves to that commit. The original source files used for Module, optimizers and Linear match the installed full files byte for byte.
Executed the exact section code, and its specified one-line exercise insertion before list construction. Outputs are 136 and 976; identity checks show the optimizer, update list and named parameters refer to the same tensors. All original-code gradients are None and the optimizer has no initialized state: the displayed snippet has not calculated a loss, backward or step.
For a bounded mechanism probe, synthetic feature tensor arange(32)/32, shape [1,2,16], passes through the real image_projector and the real frozen TinyLM. Loss is the mean of squared [1,2,264] logits, denominator 528; this is not a language-quality objective or benchmark. In eval mode, backward still gives projector gradients. Backward alone changes no weights; a single AdamW step changes the two projector tensors and preserves all 32 frozen tensors. Wrapping the complete language call in no_grad removes that gradient route. Separate fresh models check that late unfreezing makes 976 trainable elements but leaves 136 in the existing optimizer, so language-block gradients are populated but its parameter delta remains zero. Building an all-parameter optimizer first, then freezing, retains 8296 optimizer elements while only 136 remain grad-enabled.

## Unresolved material issue: categorical opening prerequisite

The opening says: 「參數需允許計算梯度，還要被優化器收錄，最後實際執行更新。」 This presents gradient permission as a prerequisite for a parameter value to be updated. In ordinary fresh autograd training it is the prerequisite for accumulating *new* gradients, but it is not checked by AdamW.step. Official Adam._init_group at lines 150–156 filters on p.grad is not None; Adam.step at 229–270 iterates stored groups and uses that list. AdamW inherits this implementation. Module.requires_grad_ at 2953–2955 only changes the flag; it does not clear existing gradients or optimizer groups.
The actual counterexample creates p=1, calculates (p**2).backward() (grad=2), then p.requires_grad_(False), then AdamW([p],lr=.001).step(). The parameter becomes 0.998989999294281 with requires_grad=False and retained grad=2. No gradient was manually assigned. This is directly relevant to the lesson's phase-change advice and its all-parameters optimizer followed by freeze case.
Requested correction: distinguish permission to accumulate new gradients from consumption of an already stored .grad. Qualify the opening as the normal fresh-gradient autograd workflow, and make the phase-change advice explicit that frozen parameters must be excluded from the optimizer or have their existing gradients cleared to None when retaining the optimizer. Rebuilding a correctly filtered optimizer is already a valid strategy, and the current demonstration itself is correct.

The zero/None prose is correctly conditional: a gradient of zero can be genuine local insensitivity, while None may indicate no received gradient. The bounded check observes AdamW decoupled weight decay changing the zero-gradient p from 1 to 0.9999899864196777, while a None-gradient unused parameter stays 1. Therefore zero must not be interpreted as a promise of no update; the lesson does not make that promise. Likewise .001 is the learning-rate scale, not a claim that every coordinate delta equals .001.

All raw downloads and temporary research remain ignored. Durable evidence includes reviewer analysis, HTTPS URL/version/locator and retrieval-SHA metadata, original lesson/prerequisites, reviewer code and real stdout/stderr. No old verdict or author notes were read; any preexisting assigned report was copied as unread bytes before replacement. No GPU work, long training, installation, dataset or model download, empirical-quality replication or source edit was performed.
