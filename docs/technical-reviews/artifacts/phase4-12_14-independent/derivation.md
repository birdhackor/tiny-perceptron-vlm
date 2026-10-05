12.14 independent derivation

Reference 我不吃辣 is [我, 不, 吃, 辣], four Unicode code points.
Prediction 我不吃拉 changes only 辣 to 拉: one substitution, minimal distance 1; CER=1/4=0.25 exactly.
Prediction equal to reference: distance 0, CER=0/4=0 exactly.
Reference 不要加辣 and prediction 要加辣: delete 不; distance 1, denominator4, CER0.25. This small numerical error removes the negation, so CER does not encode requirement satisfaction.

Agreement does not entail correctness. Let requirement forbid spicy food and let both candidate answers be 加辣火鍋. Their strings are equal, but equality supplies no recipe/requirement judgement. The helper intentionally returns None for that absent judgement.

If original requirements 不要加辣 and 要加辣 both reach the text core as recognized 要加辣, and the same prior history is supplied, then the downstream inputs are identical. Sharing the downstream program alone supplies no distinguishing bit for those opposite requirements. This does not say an LLM can never guess a transcription mistake; it shows there is no guarantee of recovering the original condition from that information alone.

History is an input record, not evidence that a particular learned model will obey it. The bounded UI test stubs generation and checks only that the next call receives the prior user condition and assistant response.
