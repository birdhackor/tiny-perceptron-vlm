# Original owner V4 reinspection of 13.12

The literal reference-stage scope issue is resolved in the actual current
source. Current paragraph at `course/chapters/13.md:394` now says:

> reference對照PPO開始時的策略（通常已完成示範微調），整段PPO期間可以固定。

This explicitly names the PPO stage rather than the beginning of every part
of posttraining. The distinction agrees with current 7.17, which includes SFT
in posttraining, and with current 13.4's reference after demonstration tuning.
Current 13.15's supervised-before-PPO context is unchanged. Necessary 13.14
context now says the reference preserves the policy after demonstration tuning
at PPO start. Its diagram now labels the reference “PPO起點” and fixes it
during “整段PPO期間”; I actually rendered the current SVG with Inkscape and
opened its PNG. This checks relevant context, not another owner's verdict.

The new source supplement explicitly links
https://arxiv.org/pdf/2203.02155v1 . I personally reopened that exact original
InstructGPT PDF's first-page v1 identifier; section 3.1 Steps 1–3 on PDF page 6,
extracted text 293–316; and section 3.2 RL paragraph/equation 2 on PDF page 9,
text 484–507. It trains a supervised policy before PPO, and pi_SFT is the
reference in equation 2. The supplement's claim and link/version agree.

I also reopened PPO arXiv:1707.06347v2 section 3 definition/equation 6,
PDF page 3, text 104–118, and Algorithm 1 on PDF page 5, text 238–247:
https://arxiv.org/pdf/1707.06347v2 . Its collecting old policy and repeated
update epochs remain distinct from the fixed PPO starting reference.

After AST location I personally reread the unchanged original experiment
method at 273–290 and 326–350. Its supervised updates 278–287 occur before
frozen reference copy 289 and PPO copy 326; old likelihoods are captured
inside each rollout. Fresh static inspection asserted that order without
initializing or training models. Original method SHA:
`862dc12680fee8374679d52a1cc55a2dd325a7aebbf634668731fa7def08aa04`.

The original fence is byte-identical, SHA
`34d5d22dca97553c3d589942bc73704f2fad2d4e93473bdf0122ddb8a9a79ed2`.
The unchanged helper, original primary snapshots and previous CPU evidence
were hash checked. The prior measured pointers for ratios, axes, gradients,
clone/history and the small-log illustration were reopened. These original
executions are explicitly reused, not described as newly executed. No new
numeric example or training result was introduced by V4, so no unrelated
numeric or training run was performed.

Current whole section was read, including the new supplement, together with
complete necessary current 13.4, 13.14, 13.15 and 7.17. Their unrelated numeric
or empirical statements are not included in this narrow stage-scope verdict.
The subject section has no figure; the contextual diagram was newly checked.
The final built V4 page theme was not independently rendered in this narrow
follow-up, and prior site screenshots retain their historical source scope.

Prior REVISE canonical SHA
`8f44ad28bd8d64913afed5b8e932a7617af6f2a5f6cec7ff8c7ef36f10da43e3`
was saved opaque before inspection. All 146 preexisting PASS, REVISE,
failed-checker and proof/history files remain unchanged. The old problem and
its earlier checker failure remain traceable; this is a new owner reinspection
of new source bytes, not a rewritten old PASS.

Current source SHA:
`4e52b0f5e6afa038e0202979e85de86831e39746eca746a3fc8deb8b78fc4ffc`.
Verdict: pass. No unresolved source-stage dependency remains for 13.12.
