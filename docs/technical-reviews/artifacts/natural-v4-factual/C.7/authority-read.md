# Original-authority inspection record

Reviewer `/root/v4_review_coordinator/factual_v4_c_7`, accessed 2026-10-04.
Full third-party PDFs, HTML, and extracted text remain only under ignored
`outputs/natural-v4/factual-research/C.7/`; `authority-retrieval.json` records
actual successful and unsuccessful retrievals and successful byte hashes.

* Williams, *Simple Statistical Gradient-Following Algorithms for
  Connectionist Reinforcement Learning*, Machine Learning 8 (1992),
  publisher PDF DOI `10.1007/BF00992696`, section 4 pp.233–236,
  unnumbered update before Theorem 1 and equations (9)–(10): reward minus
  a baseline multiplies the derivative of log action probability; the
  baseline must be independent of the current sampled output conditional
  on weights and input. Theorem 1 is about the expected gradient direction,
  not success of each finite sample. Section 8.3 pp.245–246 discusses
  variance-minimizing baselines; mean reward is not universally optimal.
  This supports the lesson's qualified variance statement and prior-batch
  moving baseline, not a guarantee from its fixed-action illustration.

* PyTorch v2.14.1 original `_tensor.py`, lines 566–625: `backward` uses the
  chain rule and accumulates leaf gradients. Official 2.14 `Tensor.detach`
  docs state that the detached result does not require gradients and shares
  storage. The lesson uses it only for the before comparison. Official
  2.14 `functional.log_softmax` docs describe log-normalized probabilities
  along the selected dimension. Original v2.14.1
  `autograd/grad_mode.py`, lines 20–85, documents `no_grad`: arithmetic
  results do not require gradients; factory functions are an exception,
  irrelevant to the arithmetic update in this lesson.

* PyTorch v2.14.1 original `distributions/categorical.py`, lines 59–82,
  144–163: scores are normalized using logsumexp; sample uses multinomial,
  log_prob gathers the selected normalized score, and entropy computes
  minus the sum of p times log p. Official 2.14 Linear docs give y=xA^T+b,
  trainable weight/bias, and last-axis feature shapes. This supports the
  actual 18→48→17 repo architecture. It is not an autoregressive LM.

* Loshchilov & Hutter, *Decoupled Weight Decay Regularization*,
  arXiv:1711.05101v3 (2019-01-04), section 2 and Algorithm 2 pp.2–3:
  adaptive first/second moments and weight decay are separate. The
  publisher uses a decay parameterization distinct from PyTorch's
  lr-times-decay convention. Exact current software behavior was checked
  in PyTorch v2.14.1 original `optim/adamw.py`, lines 20–60 and its
  algorithm docstring: `decoupled_weight_decay=True`; default decay .01,
  betas (.9,.999), epsilon 1e-8. No optimizer superiority is inferred.

* TÜLU 3, arXiv:2411.15124v3, section 4/4.1 p.15 and section 6
  pp.30–31, Figure 18 and equations (7)–(8): SFT learns prompt/completion
  demonstrations; RLVR samples policy completions and rewards verification
  of correctness. TÜLU's particular optimizer is PPO with a KL term; this
  lesson separately and explicitly uses finite-action REINFORCE, so the
  source supports RLVR terminology and verification, not identity of their
  optimization algorithms. DeepSeek-R1 arXiv:2501.12948v1 section 2.2.2 p.6
  was also read: math deterministic results and code test cases supply
  rule-based reward, with format requirements. It corroborates this
  limited usage rather than proving every verifier is reliable.

* Turpin et al., arXiv:2305.04388v1 (2023-05-07), abstract and introduction
  pp.1–2, Table 1: plausible chains of thought can omit causal effects of
  input biases. This supports the need for separate faithfulness evidence,
  not a claim that every chain of thought is unfaithful.

* Python 3.13 official builtins `round`: ndigits controls decimal display;
  midpoint ties go to the even choice, with binary floating point caveats.
  The six-decimal outputs here are not midpoint ties and the list
  comprehension constructs a display list without changing model tensors.

* Git LFS original `docs/man/git-lfs-install.adoc` and
  `git-lfs-pull.adoc`, main snapshots retrieved 2026-10-04: `--local`
  limits configuration to the repository; `pull` fetches/checks out
  current-ref LFS objects; `--include` filters paths and empty `--exclude`
  clears that exclusion. Neither mutating Git command was run. The
  experiment's asset listing and help were genuinely executed, while
  the long optional complete schedule was inspected, not rerun.

Current prerequisite bodies were actually read and are recorded in
`prerequisite-read.json`; their raw original bodies are retained in ignored
research. No previous technical or reader verdict was used. The assigned
old technical report was copied unread before replacement. The initial
CPU helper had a syntax error; its actual stderr is preserved separately,
then the corrected helper executed successfully. That helper mistake was
not a factual defect of the lesson.
