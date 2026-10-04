# Independent derivations — 7.18

Reviewer: /root/v4_review_coordinator/factual_v4_7_18; fresh. Reviewed 2026-10-04.

1. Arithmetic: 1+2=1+1+1=3. The demonstration 3 has the correct value, and the literal 4 differs from 3. The code assigns reward 0 by hand; 0 is a chosen score for this example, not a universal reward rule.
2. Format: only digits are requested. 3 meets that rule. 答案是3喔！ states the same correct arithmetic value, but includes Chinese characters and punctuation, so violates the format. Preference compares suitability for this prompt, not arithmetic correctness alone.
3. Reward distinction: choosing 0 for one wrong answer does not specify scores for all answers or other tasks. A complete reward rule must identify truth, parsing/format rules and intended task; its numerical output does not imply real-world quality.
4. Loss versus completion: suppose the true answer probability increases from 0.3 to 0.4 and the sole wrong answer falls from 0.7 to 0.6. Since log is increasing, -log(0.4)<-log(0.3); loss declines but argmax still selects the wrong answer in both cases. Declining loss does not entail task success.
5. Changed prompt: 用一句話解釋1+2 requests an explanation, e.g. 一個加上兩個，一共有三個。 Neither 3 nor 答案是3喔！ explains combining one and two. Exchanging chosen/rejected retains the missing explanation, so a new demonstration and preference criteria are needed.
6. DPO reference: Eq.(7) of 2305.18290v1 compares policy/reference log ratios for the same chosen/rejected answers. Initial policy equal to reference makes each ratio 1 and log ratio 0; a fixed reference lets changes represent changes relative to that baseline. This is standard DPO here; reference-free variants are outside the lesson.

These are independent hand calculations/logic. The separate execution transcript records the actual current snippet. No paper benchmark, training, GPU work or model generation was reproduced.
