# Independent C.1 factual inspection

Reviewer: `/root/v4_review_coordinator/factual_v4_c_1`; fresh factual context.
Read the complete raw C.1 section, all intervening blank lines and its complete before-first-section introduction. The assigned hashes match. The previous assigned report was copied byte-for-byte to `previous-report.unread.json` without parsing or reading its contents.

Read the current complete necessary prerequisite sections 1.14 and 7.11, plus the linked temperature/NLL explanations 1.15 and 1.8 and C.2's referenced counterexample. Their inspected section hashes are respectively:

- 1.14: `bcc81e3a8f25eb8ba833e74c13a3a1b0decbd5926b41ab91c7defbd6d7b5fa38`
- 7.11: `4622ff4e08f4756beaba1cd2efd5ad46ec0c477f65f49a6a31c17a37de98a168`
- 1.15: `4073bbabc84d9045c7791db2ed8f59584b70a69c8ce5007545ede1f197e0c9ce`
- 1.8: `c7ad6f245be1f0ebeb314d2985c81a7468d5326758ce34394435546c697ea574`
- C.2: `6abf752dfad6dff873d0d90bdc85648865e7e713330ebc6a0965330bc3d69167`

C.1, its introduction, and these needed prerequisite sections contain no SVG references. No figure render is needed for their factual inspection. The C.7 link is navigation to the optional full-run entry, not a prerequisite for the C.1 claim audit. The actual experiment implementation was inspected directly without running its training schedule.

## Original authorities actually inspected

Retrieved exact-version originals into ignored `outputs/natural-v4/factual-research/C.1/`; the small retrieval receipt records original HTTPS URLs, retrieval date, byte counts and SHA-256. No complete third-party text or figure is copied into this evidence directory.

- Wei et al., arXiv 2201.11903v6, dated 10 January 2023: section 2, pages 2–3, describes intermediate steps, potential additional computation, debugging and the unresolved task of fully characterizing underlying computation. Section 3.1, page 3, separates direct-answer and step demonstrations and evaluates on independent benchmark problems. Its published large-model gains do not establish gains for this course's tiny SFT run.
- Ouyang et al., arXiv 2203.02155v1: section 3.1, page 5, explicitly separates supervised demonstration learning, comparison/reward-model fitting, and PPO updates. Section 3.5, page 6, describes supervised fine-tuning on labeler demonstrations. This supports the training mechanism and distinguishes inference from updates; it does not prove any particular arithmetic score.
- Turpin et al., arXiv 2305.04388v2, dated 9 December 2023: introduction, pages 1–2; section 2, page 3; and section 6's limitations, page 10. Predictions are conditioned on the generated reasoning, but plausible explanations can omit influential input features and can rationalize wrong answers. The paper's counterfactual tests detect failures of faithfulness, rather than establish complete internal traces. This supports the lesson's cautious possibility statements, not the claim that every generated step is unfaithful.
- Hinton, Vinyals and Dean, arXiv 1503.02531v1: section 2, equation (1), pages 2–3. Temperature divides logits inside softmax; larger positive temperature softens the distribution. Only this probability transformation is used here, not distillation's training objective.
- PyTorch's official `torch/nn/modules/loss.py` at commit `5c4886908584029761b579af026dcfb627c84070`: `CrossEntropyLoss`, lines 1200–1243, index-target formula, ignore mask and sum/mean reductions. The fetched file SHA-256 is `415e4ccbbab63b9cc2b79f09a338cca6fa191c07138a4fb419e3c841c4748ae0`, exactly matching the installed file in the actual `2.14.1+cpu` runtime. The stable documentation URL returned HTTP 403; the matching official implementation resolved that retrieval obstacle. No API claim relies on an unread stable page.

## Derivations and bounded execution

The exact C.1 code block was executed, with separate stdout and stderr. It emits answer lengths 1 and 47. The proposed last-digit substitution emits 47: changing a byte-valued digit does not add or remove a token and does not check arithmetic. UTF-8 encoding of this particular Chinese answer has 14 non-ASCII characters at three bytes each and five ASCII characters at one byte each, so 14×3+5=47. Plain `encode` adds no special IDs. Three increments from 2 give 3, 4, 5; replacing the conclusion by 6 contradicts those steps. C.2's separate 2+2=5;5−1=4 example likewise has a false first equality and a true final 4.

All 6×6×6=216 ordered triples form C(8,3)=56 unordered three-number families. Reconstructing the family-first shuffle with seed 42 gives 44/6/6 families and 167/25/24 examples. No family crosses splits. Input numbers remain in the original 0–5 range: this tests held-out combinations, not a larger numeric range.

Independent response construction, exact SFT label/mask comparison, and a 900×24-draw Python RNG reconstruction reproduce 48,803 and 466,322 effective training targets, including one EOS per sampled answer. Both branches sample 21,600 training examples from 167 unique examples. The target ratio is 466,322/48,803=9.5551912792, which rounds to about 9.6. The full final-check answers contribute 51 and 510 effective targets. Full-training answer sets contain 377 and 3,605 targets, respectively; repeated sampled training totals use a different denominator.

With no class weights and integer labels, each included loss is −log softmax(z)[target]. The project uses a summed cross entropy then divides by the number of labels not equal to −100. The deterministic float64 probe gives sum 1.5062182531124901 over two included positions, mean 0.7531091265562451, and zero gradient at the ignored position. Full correct-answer prefixes are supplied by `render_chat` and the causal model; teacher-forced NLL does not measure free-running solution accuracy. In the official record, 220.10995483398438/51=4.315881467333027 and 69.97113037109375/510=0.13719829484528187. These round to 4.3159 and 0.1372. Recorded complete-training averages round to 0.0268 and 0.000262; their learned-model forward passes were not reproduced.

For fixed logits and positive T, p_i(T)=exp(z_i/T)/sum_j exp(z_j/T). For z_i>z_j the odds ratio exp((z_i−z_j)/T) grows as T decreases; tied logits remain tied. Entropy derivative is dH/dT=Var_p(z)/T³≥0. Thus T=0.7 is no more diffuse than T=1, with equality only for equal logits on the support. CPU probe [3,1,0] gives leading probabilities 0.93353561905 vs 0.84379473448 and entropies 0.27703008466 vs 0.52426661673. This is an inference transformation, not a parameter update.

## Official fixed-run audit and its limits

The authoritative project record is `docs/course-experiments/results/reasoning.json`, completed revision `910aebc6419c9fc6217279a27fde9851c5cfad30`, seed 42, NVIDIA L4, PyTorch 2.14.1+cu126, Python 3.13.3. Its code registry matches the inspected current data/model/attention/common/applications files. `run_reasoning` constructs a random base, copies that state into both branches and supplies width 64, layers 2, heads 2, max length 128, 900 updates and batch size 24. `fit_lm` uses AdamW's default learning rate 0.003. It does not load a pretrained or SFT checkpoint for those branches. The shared base-state digest is identical in both recorded branches.

The record contains all single-candidate generated byte IDs, decoded answers, original questions, original operands, seed-level configuration, maximum lengths and scoring fields. Independently decoding and parsing those actual generated IDs gives 1/24 correct direct answers and 11/24 correct step answers. Therefore 13 step answers fail. Temperature is 0.7 throughout, maximum new-token caps are 8 and 48, and actual generated counts are 51 and 512 including EOS. The two runs have unequal target and generation budgets, so the lesson's one-run observation is supported; no equal-compute benefit or general reasoning improvement follows. NLL averages on different answer texts are not an interchangeable ability measure.

This is an independent audit/recalculation of official fixed experiment records, not independent GPU training, checkpoint evaluation, or benchmark replication. No training, downloaded model, new dataset, install, Git change or persistent environment change was used. No stable speed conclusion is drawn. All unresolved factual issues: none.
