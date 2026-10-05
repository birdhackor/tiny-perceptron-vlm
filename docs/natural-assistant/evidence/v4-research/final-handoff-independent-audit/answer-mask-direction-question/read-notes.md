# Own answer-mask direction question

The figure’s A/EOS alignment is correct. The visible heading `下一項目標 Y（已右移）` (SVG line18) and accessible description `已右移的目標` (line3) contain a nonblocking direction ambiguity.

I personally read the complete current20.7, its7.4 alignment prerequisite and the referenced1.3 shift explanation, inspected the current raw SVG, and personally viewed both registered original figure PNGs. The answer-mask render receipt binds SHA `f82ac52237b5ce6306f8c28be8959940ae275961643d091a9107bf1bd86970f7`, which equals the current SVG. I used the20.7 factual registry to locate originals, without adopting its verdict.

The actual toy code at `tiny_perceptron/data.py:68` returns `ids[:-1]` and `targets[1:]`. Original A is at index5 and appears in aligned Y at index4; original finalEOS at index6 appears at index5. My bounded CPU call with the repository `.venv` gives X `[1,3,89,2,4,73]`, Y `[-100,-100,-100,-100,73,2]`. The unchanged toy loss at `model.py:92–105` compares these aligned positions directly. The figure blue boxes and its first-answer sentence match this behavior.

I also read the cached original Transformersv4.57.6 `loss_utils.py:45–68`, matching the registered SHA `fa65f18b3167f45989d7a8ba86432a41d322338f9789cc89b08d5d7f20f110b7`. Lines58–60 say “Shift so that tokens < n predict n,” pad an ignored target and take `labels[...,1:]`. Original Qwen `modeling_qwen3_vl.py:1364–1366` dispatches raw labels to that loss. These are original implementation sources, not a secondary tutorial or prior review judgment. This inspection does not execute Qwen.

Taken literally as physical motion of Y, “already shifted right” is inaccurate: targets move to one earlier array index for alignment. But1.3 says the answer is the item originally to the right of the current input and calls that next-item relation右移;7.4 and20.7 explicitly give the correct slices and predictor positions. Under that convention the phrase is defensible. I therefore classify this as a wording ambiguity, not a material target-mask or numerical figure error.

Recommended optional simplification: line18 `下一項目標 Y`, or `每格要預測的下一項 Y`. If wording is changed, accessible line3 should also describe `與輸入逐格對齊的下一項目標`. The nearby20.7 line241 could similarly say `這個下一項目標的對齊關係`, although its current cross-reference leads to an exact correct example. Renaming the label to左移 would introduce a different movement convention without helping this teaching goal.

No source, code, reviewer report or previous audit file was changed. No GPU, model inference, training or network write occurred. This recommendation does not force a change or reject the ongoing frozen-source browser verification.
