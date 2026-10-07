# MTP 支持段落索引

本檔只保存研究所需短引文。原始 PDF、HTML 與全文抽取在被忽略的 `outputs/phase7-source-cache/mtp/`；來源 URL、版本、檔案 SHA-256 與本次範圍見 `source-index.json`。引用段落以原文 PDF 頁數定位，行尾斷字及空白已整理並與保存的全文核對。

這輪是技術來源研究，不是首次閱讀者重審、教材驗收或模型驗收。

## G1 — Abstract

來源：Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，PDF p.1；https://arxiv.org/pdf/2404.19737v1

> at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk.

支持：平行 future-token heads 的身分；n 包含下一 token head。

## G2 — §2 Method

來源：Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，PDF p.2；https://arxiv.org/pdf/2404.19737v1

> n independent output heads implemented in terms of transformer layers

支持：獨立的是分支；原論文分支含 Transformer layer，並共用 unembedding。

## G3 — §2 Inference

來源：Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，PDF p.2；https://arxiv.org/pdf/2404.19737v1

> while discarding all others. However, the additional output heads can be leveraged to speed up decoding

支持：丟掉額外 heads 可正常推論；加速是額外解碼機制。

## G4 — Figure 3 caption / §3.1

來源：Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，PDF p.3；https://arxiv.org/pdf/2404.19737v1

> Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale.

支持：code 任務的 300M–13B 尺度觀察，不能外推全部小模型。

## G5 — Appendix D, Table S6

來源：Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，PDF p.15；https://arxiv.org/pdf/2404.19737v1

> Finetuning LLama 2 with multi-token prediction does not significantly improve performance.

支持：既有 NTP 模型末端加 MTP 不能當作已證實的改善配方。

## G6 — §4.1

來源：Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，PDF p.6；https://arxiv.org/pdf/2404.19737v1

> We find that 2-token prediction loss leads to a vastly improved formation of induction capability for models of size 30M nonembedding parameters and below

支持：小型受控任務也有正面結果，不能把 G4 寫成小模型通則。

## G7 — §3 experiment setup

來源：Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，PDF p.3；https://arxiv.org/pdf/2404.19737v1

> when we add n − 1 layers in future prediction heads, we remove n − 1 layers from the shared model trunk.

支持：論文的等參數配對不是在原模型免費加數個 heads。

## G8 — §3.2 Faster inference

來源：Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，PDF p.3；https://arxiv.org/pdf/2404.19737v1

> We implement greedy self-speculative decoding

支持：3× 論文加速依賴實際 greedy 自我推測解碼及量測。

## D1 — §2.2

來源：DeepSeek-V3 Technical Report，arXiv:2412.19437v2，PDF p.10；https://arxiv.org/pdf/2412.19437v2

> we sequentially predict additional tokens and keep the complete causal chain at each prediction depth.

支持：DeepSeek 不是 G1 的平行獨立分支。

## D2 — §2.2 MTP Modules

來源：DeepSeek-V3 Technical Report，arXiv:2412.19437v2，PDF p.10；https://arxiv.org/pdf/2412.19437v2

> our MTP implementation uses 𝐷 sequential modules to predict 𝐷 additional tokens.

支持：D 計額外模組；k=1 接下一 token 嵌入，預測再下一個。

## D3 — §2.2 MTP Training Objective, Eq.25

來源：DeepSeek-V3 Technical Report，arXiv:2412.19437v2，PDF p.11；https://arxiv.org/pdf/2412.19437v2

> we compute the average of the MTP losses across all depths and multiply it by a weighting factor

支持：跨深度平均乘 λ；Eq.24 的每深度分母是 T。

## D4 — §2.2 MTP in Inference

來源：DeepSeek-V3 Technical Report，arXiv:2412.19437v2，PDF p.11；https://arxiv.org/pdf/2412.19437v2

> during inference, we can directly discard the MTP modules and the main model can function independently and normally.

支持：訓練輔助分支可以不部署；不能因此聲稱推測解碼已啟用。

## D5 — §4.2 Model Hyper-Parameters

來源：DeepSeek-V3 Technical Report，arXiv:2412.19437v2，PDF p.22；https://arxiv.org/pdf/2412.19437v2

> The multi-token prediction depth 𝐷 is set to 1

支持：實際 V3 是下一 token 加一個額外 token，圖中的 D=2 是機制示意。

## D6 — §4.5.1

來源：DeepSeek-V3 Technical Report，arXiv:2412.19437v2，PDF p.26；https://arxiv.org/pdf/2412.19437v2

> keeping the training data and the other architectures the same, we append a 1-depth MTP module

支持：15.7B 與 228.7B MoE 的 ablation 支持相應規模的多數 benchmark；非 5.45M 成品。

## D7 — §5.4.3

來源：DeepSeek-V3 Technical Report，arXiv:2412.19437v2，PDF p.35；https://arxiv.org/pdf/2412.19437v2

> the acceptance rate of the second token prediction ranges between 85% and 90%

支持：作者在 V3 生成主題的實測；不是本專案驗收或一般下限。

## D8 — §5.4.3

來源：DeepSeek-V3 Technical Report，arXiv:2412.19437v2，PDF p.35；https://arxiv.org/pdf/2412.19437v2

> delivering 1.8 times TPS (Tokens Per Second).

支持：加速比例是原報告結果，沒有充分硬體/計時細節可據以保證本專案。

## L1 — §2.1

來源：Fast Inference from Transformers via Speculative Decoding，ICML 2023 / PMLR 202:19274–19286，PDF p.2；https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf

> use the target model Mp to evaluate all of the guesses and their respective probabilities from Mq in parallel

支持：草稿並不直接成為答案，需要主模型驗證。

## L2 — §2.3 / Algorithm 1

來源：Fast Inference from Transformers via Speculative Decoding，ICML 2023 / PMLR 202:19274–19286，PDF p.3；https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf

> If x1 is rejected, we discard the computation of x2

支持：首個拒絕後，後續依賴錯誤前綴的候選計算不能保留。

## L3 — §3.3

來源：Fast Inference from Transformers via Speculative Decoding，ICML 2023 / PMLR 202:19274–19286，PDF p.4；https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf

> the value of c depends on the hardware configuration and software implementation details.

支持：接受率以外還有提議及驗證成本，不能由 head 數推出速度。
