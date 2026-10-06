# 13.12 original-owner narrow scope finding

Verdict: revise. This is a new finding on the fixed current source, not a
rewriting of the accepted PASS or its original trace.

Exact target quote, `course/chapters/13.md:394`, section-relative line 22:

> old與reference也不同：old對照本輪收集那一刻，下輪重新收集時換新紀錄；reference對照整個後訓練起點，可以全程不動。兩者都叫「舊模型」會混淆用途。

The old-policy distinction and the possibility of a fixed reference are
supported. The remaining substantive scope problem is **整個後訓練起點**.
The chapter uses posttraining to include SFT: current 7.17, chapter line 624,
explicitly says “接在底座後的學習叫後訓練，SFT是其中一種”. Current 13.4,
chapter line 125, instead identifies the reference as “完成示範微調的起點”.
Current 13.15, chapter line 564, describes initial supervised demonstration
training before PPO and connects this to the LLM SFT stage. Those are
compatible with a reference at the **PPO stage start**, not the start before
every part of posttraining.

Original authoritative support was personally reopened:

- Ouyang et al., InstructGPT, arXiv:2203.02155v1, original PDF first-page
  version identifier. Introduction and Figure 2 caption on PDF pages 2–3;
  section 3.1 Steps 1–3, extracted text lines 304–316; section 3.2 RL paragraph
  and equation 2 on PDF page 9, text lines 484–507.
  https://arxiv.org/pdf/2203.02155v1
  The pipeline first trains a supervised policy, then optimizes that policy
  with PPO; the KL reference in equation 2 is pi_SFT.
- Schulman et al., PPO, arXiv:1707.06347v2, original PDF section 3 definition
  and equations 6–7, PDF page 3, text 104–130; Algorithm 1, PDF page 5,
  text 238–247.
  https://arxiv.org/pdf/1707.06347v2
  Old is the collecting policy and refreshes after the update epochs.
  This paper does not identify a reference as the start before all LLM
  posttraining stages.
- Current original `scripts/course_experiments/posttraining.py`, full SHA
  `862dc12680fee8374679d52a1cc55a2dd325a7aebbf634668731fa7def08aa04`.
  After AST location, personally read 256–295 and 326–352. The supervised
  update loop is 278–287; frozen reference is copied at 289; PPO policy is
  copied from that reference at 326; rollout-specific old values are captured
  within the loop starting 335. The bounded static inspection asserts this
  exact stage order without initializing or training any model.

Necessary current 13.14 context was also read, including its original role
diagram. Its title establishes a PPO setting, but the reference box says
“整段後訓練都不更新” and the paragraph says “reference保留整段後訓練起點”.
This repeats the broad term rather than explicitly defining posttraining here
to exclude SFT. I rendered that exact SVG with Inkscape and opened the PNG.
This is context evidence; it is not a technical verdict for another owner's
section.

The current necessary context therefore contains the right SFT-to-PPO start
and a conflicting broad scope. A reviewer can silently restrict “後訓練” to
the preference/PPO stage, but the source itself says “整個”, while its prior
definition includes SFT. This can make a reader choose the pre-SFT base as
the reference and change what the KL deviation is measured against.

Necessary suggestion for the root author: replace that part with a precise
stage boundary, for example:

> reference對照PPO開始時的策略（通常已完成示範微調），整段PPO期間可以固定。

Keep the old/reference distinction. No mathematical example, original fence,
training result or other claim needs to be deleted. Related 13.14 wording can
be sent to its own owner if the root author changes it. I did not modify any
lesson, figure or another owner's report.

The accepted report and every previous artifact remain opaque and unchanged.
Current section SHA is
`10591ac40861f01e2f54ee4542070d2de7de45faea502c6528f11decd0ac8575`.
Unchanged original code/numeric artifacts retain their exact earlier hashes
and support only their already verified arithmetic/graph/storage behavior.
No GPU, training, model/data download, model reevaluation or new broad search
was run. This narrow finding uses method stage ordering, not a model score.
