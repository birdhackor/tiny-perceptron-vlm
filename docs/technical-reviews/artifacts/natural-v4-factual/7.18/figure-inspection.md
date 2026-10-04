# Personal rendered-figure inspection — 7.18

Reviewer: /root/v4_review_coordinator/factual_v4_7_18; fresh; 2026-10-04.
Original: course/figures/posttrain_signals.svg. Inkscape 1.4 rendered posttrain_signals.png at 1230 by 735. I personally viewed it with view_image, including after the recorded rerender.

Top row left to right: one good answer, two-answer comparison, score after answering. Left card requests only digits for 1+2 and demonstrates 3, matching section lines 3,8,16. Center correctly prefers 3 to the longer phrase for this prompt; it does not claim the longer phrase is arithmetically false.

Left downward arrow leads to SFT/increasing demonstration probability. Middle downward arrow leads to DPO/direct relative preference. Right downward arrow leads to PPO and other update methods/feedback-based choices. A separate branch leaves the center card's right edge, bends down and then upward into the scoring card's lower-left side; its arrowhead enters the scoring route. Nearby labels say first teach a scorer, then score new answers. This is preference-to-reward-model, not a mandatory sequential arrow; it is separate from the right downward arrow.

Scoring card allows verification scores. Footer distinguishes typical human-feedback RLHF from program verification, and states routes can be selected/combined without all being required. All labels/arrowheads are visible; no numerical axis, performance scale or unstated improvement appears. Renderer emitted two wrapper warnings, exited 0; the viewed output is complete. Original SVG unchanged.
