# 9.8 fresh reading and predictions (saved before execution)

Reviewer task: /root/v4_review_coordinator/reader_cleanup_9_8

Actual reading order: reader-contract.md; the complete 9.8 snapshot; complete 9.2; complete 9.7; complete 9.6; complete 8.3. The 9.2 and 9.7 sections are explicitly named prerequisites. I followed 9.6 to resolve how matches and EOS are counted, and 8.3 to understand the named addition baseline and unequal numbers of supervised answer positions. No other chapter/section, experiment report, author note or reader report was opened.

I copied the existing 9.8 report bytes to prior-report-unread.json without reading them. I did not use that report to decide my verdict.

## Before running the short example

Prediction: the outer loop first visits train, then test, because the dictionary is written in that order. Each currently has one prompt. Each line prints its split, its own prompt, an arrow, and the same expected sentence: 資訊不足，請提供數量或可看清的圖片。 The final line is 訓練問句與測試問句相同 False. Thus there are three output lines. No model is called or trained. The assigned expected string cannot show whether a model answers correctly. Comparing two strings only establishes unequal full text; their template membership still needs a reader's judgment.

## Before running the exercise

Prediction: adding 只看外觀能知道盒內球數嗎 to test produces one train line and two test lines, followed by the same False comparison, for four output lines. The new test line still asks for quantity or a usable picture because appearance alone supplies no count or appearance-to-count rule. I keep the new prompt out of train. The existing last equality check compares only the first test prompt to the first train prompt; I check the full output separately to count both test lines.

## Reading comprehension

The question is whether the learned response follows the available-information rule when wording changes, or only reproduces familiar mappings. A family split can reduce near duplication, but the actual experiment with eight rule-combination families did not fully hold out the main wording templates. The 136 deduplicated behavior questions split into 102=6×17 training, 17 validation and 17 final questions; validation/final are never used for weight updates and are disjoint. The same addition baseline is copied, one version uses only 102 behavior questions, the other also has 49 addition questions (151 total). Both run 900 updates with seed 42, which makes the arrangement repeatable but does not equalize supervised answer positions. The 15/17 and 16/17 behavior matches have a separate denominator from each 0/7 addition result. Baseline already scoring 0/7 means this cannot demonstrate destroyed prior addition competence. Mixed model validation is 14/17. Keeping the mixed model fixed and rewriting six final prompts gives 0/6 exact matches with stopping markers present. In the document example, saying red instead of adversarial pink still fails the original job of finding green. The six rewrites involve only two limited forms, not arbitrary wording or multiple turns. A per-turn trace preserves which input caused the failure.

9.6 explains that a match is the whole generated answer-ID sequence agreeing with the target sequence, and EOS is a separate stop marker. It also explains why proper clarification counts among the normal-completion cases. 8.3 supplies the same addition baseline's already-zero seven-question score and shows why longer answers have more supervised positions. I could understand the stated scope with these direct links; further links in those prerequisites were not necessary.

## Figure inspection

The complete 9.8 and the four necessary linked sections I read contain no image/SVG references. No SVG was rendered or viewed, and no render/view is claimed. figure_sha256 is consequently {}.

## Tentative assessment

The section supplies a question, a runnable design example, line-by-line explanation, clearly scoped empirical results and a checkable exercise. The explicitly linked 9.6 resolves the scoring and EOS terms. I have no remaining readability blocker after following these links. This is a readability assessment, not a factual audit or a rerun of the model experiments.
