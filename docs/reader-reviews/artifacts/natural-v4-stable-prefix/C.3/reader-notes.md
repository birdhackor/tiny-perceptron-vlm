# C.3 fresh reader evidence

Reviewer: /root/v4_review_coordinator/reader_stable_c_3. I used the perspective of a mathematically comfortable high-school / early-university reader seeing this project for the first time. These are my own explanations after personally reading the four complete assigned/necessary sections in read-order.json. No old review was consulted.

## My explanation of the question and small example

The question asks whether making several candidate answers gives us more opportunities to have a correct answer available. It separates this from the later task of picking the correct candidate. Under the declared independent, same-question, same-distribution assumption, one answer is wrong with probability 1-p. All n answers are wrong with probability (1-p)^n. The complementary event is at least one correct answer, hence 1-(1-p)^n. For p=0.3 and n=2, the two-failure probability is 0.7×0.7=0.49 and its complement is 0.51. Independence is what licenses the multiplication. Neither independence nor the value 0.3 is established by the demonstration program.

p denotes the success probability of one complete answer to this fixed question, n the number of generated candidates; 1 is the probability of the full set of possibilities; ** is Python exponentiation. The mathematical coverage values for n=1,2,4,8 are 0.3,0.51,0.7599,0.94235199, with the last rounded to 0.942352. These are probabilities under the assumption, not counts and not measured model scores.

The list [3,4,3,5] is supplied by a person, with 4 supplied as the known truth. It contains 4 exactly at its second position. That gives True for membership even though selecting its first answer would return incorrect 3. The list need not have been sampled from any distribution. Repeated 3 values alone cannot reveal whether actual random trials are dependent. For a fixed question an independent sampler can give a high probability to the same wrong answer on every draw.

## My explanation of input, code, and output

There is no stdin, model checkpoint, network request, or training step. The inputs are p=0.3, the tuple (1,2,4,8), candidates, and truth 4 inside the membership expression. The for loop assigns n successively, evaluates coverage, and prints the Chinese labels with the current n and round(coverage,6). round changes the display precision, not the sampling model. Then the manually entered list is printed and 4 in candidates returns a Boolean. It does not generate candidates, rank them, pick a winner, or calculate an empirical p.

I wrote exercise-prediction.txt before either run, then extracted the exact Python fence to baseline.py and changed only the candidates assignment in exercise.py. execution-record.json contains the actual .venv Python command, cwd, return code, stdout, and stderr. Both runs returned 0 and the output comparison passed. The four coverage rows remained byte-for-byte identical; only the final list and Boolean changed from [3,4,3,5]/True to [3,3,3,3]/False. This validates this short example and the requested exercise only.

## Conditions, terms, and boundaries I can explain

Coverage means the set includes a correct answer. pass@k is introduced as an evaluation label that must specify its estimation procedure and sampling settings. The text defines the special measured oracle coverage as checking each sampled set against known arithmetic truth. It does not promise a deployable selector that already knows that truth. For the cited estimator, n is the total available candidate count, c the successful count, and k the evaluated subset size; when n=k there is only the full set, so the outcome is 0 for c=0 and 1 otherwise. I can follow this particular logical claim without having read or independently verified the linked original paper; I have not learned the general n>k estimator from this section.

1.15 gives the needed meaning of temperature: divide candidate scores by a positive temperature before forming sampling probabilities. At 0.7 the distribution is more concentrated than at 1 for the same scores; this does not change learned parameters. Its greedy/sampling distinction lets me understand deterministic generation. A deterministically wrong fixed question has p=0 and repeated calls remain wrong. Changing the next prompt based on prior output or reusing the same random sequence means the independence premise needs a fresh check.

A different question can have a different p. As my own mathematical explanation of why an average p is unsafe, imagine two equally weighted questions with p=0 and p=1. The average single-answer accuracy is 0.5, but with two candidates their mean coverage is still 0.5, whereas putting 0.5 into 1-(1-p)^2 would give 0.75. This is my reasoning example, not a course experiment and not an extra executed model test.

Token is defined locally as a unit of processed/generated text. C.1 explains the relevant budget contrast and its ByteTokenizer example; I did not need to study model architecture to follow this comparison. More candidates normally mean more generated tokens/work, while concurrent hardware can change elapsed time. Fixed answer length, warm-up, and first-run overhead matter when comparing timing. The present short program does not measure any of those costs.

## My explanation of the formal table and what it cannot establish

C.1 provides the two separately trained branches and data split. They start as copies of the same random initial parameters, then receive different targets; they do not end with the same weights. The short-answer target for (1+2)+3=? is 6. The steps target is 1+2=3;3+3=6;answer=6. These are human training targets, not guaranteed actual model responses. C.1 says the test asks about withheld combinations of numbers from 0 through 5, rather than a new range of numbers. Equal 900 updates is not equal target-token exposure: the step answers are longer.

The C.3 table has 24 held-out questions as every denominator. For k=1,2,4,8, the short branch has 1,1,2,1 sets with a correct final answer; the step branch has 11,13,13,14. k here is the number of candidates for each one question at each sampling setting. Each k uses a freshly sampled batch, so a k=8 set is not guaranteed to include the successful answer seen in a k=4 set. This explains why the empirical short branch can fall from 2/24 to 1/24 despite the independent fixed-p probability increasing with n. There are 24×(1+2+4+8)=360 candidates per branch, but 24 distinct tested questions, not 360 distinct questions.

These counts, temperature 0.7, raw identifiers, and retained candidates are statements of an existing course experiment. I did not reproduce them, inspect the raw result JSON, train either model, execute HumanEval, or test a real answer selector. Readability of this section and agreement of its tiny Python example do not verify those existing measurements, the factual accuracy of the whole course, or general deployment performance.

## Necessary prerequisites and remaining readability concerns

W.4 supplied probability and the independent-event multiplication; 1.15 supplied generation rules and temperature; C.1 supplied the branch design, held-out combinations, and unequal text budgets. I did not need to follow their onward links to understand C.3. There were no core terms left without enough explanation for this reading task. “原始ID” is not expanded as an English abbreviation but is understandable here as stored candidate identifiers; its storage format is not needed for the coverage reasoning. Detailed weight updating, layer/head architecture, and the general pass@k estimator are outside this section's demonstration, not hidden steps required to run its program.

I found no blocking missing prerequisite, unexplained numerical step, inconsistent program explanation, or image/text obstacle. C.3 and the three actually used prerequisite sections contain no image references or SVG. Accordingly, figure-audit.json records no render and no view_image call, and figure_sha256 is {}. This is an actual absence finding, not a claim to have inspected nonexistent figures.

## Exercise reasoning and verdict

Changing only candidates to [3,3,3,3] removes the truth 4, so membership must be False. It has no effect on p or n and therefore no effect on the preceding four formula rows. It also does not transform the entered list into a random experiment or re-estimate single-trial p. My prediction and the actual run agree. The exercise directly checks the distinction between theoretical chance and observed membership, so I judge C.3 pass for this fresh-reader readability review. No source amendment is requested.
