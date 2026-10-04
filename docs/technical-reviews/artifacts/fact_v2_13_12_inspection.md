# 獨立審閱範圍與親讀記錄

reviewer_task: /root/integration_technical_coordinator/fact_v2_13_12
reviewer_context: fresh
日期：2026-10-04。完整讀取 docs/technical-review-guide.md、本節13.12、明示必要前置13.11與13.4。不讀舊 review 結論或作者論證／歷史，不猜 author_tasks，不委派其他代理。不修改教材或環境。

原論文直接由 outputs/posttrain-design/ppo-reference 的原 PDF 使用 pdftotext -layout 重新擷取，版本由第一頁 arXiv 標識核實。原 PDF 的 bytes SHA、擷取 command 見 original_manifest.json；來源摘要或舊代理筆記不當作權威。

- PPO arXiv:1707.06347v2，2017-08-28。親讀 §2.1 的 empirical average 與 repeated LPG 限制、§3 公式6–7，及 §5 公式9–12／Algorithm1。ratio 是同一 st、at 的 new/old，採樣分母正機率；式6提高 mean(ratio×A)，不是回報品質；式7才加入裁切。Algorithm1 先 rollout、算 A，做 K epochs，再 θ_old←θ。本節是單次卡片選擇的介紹，沒有聲稱完整 token 級 PPO。
- InstructGPT arXiv:2203.02155v1，2022-03-04。親讀 §3.5（PDF8–9頁）：從SFT模型開始RL，對 π_SFT 添加 per-token KL，式2將 π_RL 與 π_SFT 分開。它支持固定起點 reference 角色；不是每輪採樣 old denominator，也沒有宣稱 KL 是絕對硬界。
- DPO arXiv:2305.18290v3，2024-07-29。親讀 §3 公式3與第4頁開始解釋：π_ref 指 initial SFT，策略也通常由SFT初始化；§4 公式4–7及 Practical Implementation 解釋 reference 初始化。參考是常見SFT設定，並非所有後訓練必須採用的唯一來源。本節寫「仍可以是」，符合此條件。
- OpenAI lm-human-preferences 原始碼 pinned cbfd210bb8b08f6bc5c26878c10984b90f516c66，親讀 train_policy.py:149–154 的 reference KL、180–194 的 rollout重用、280–299 的採樣和 reference分數、319–355 的 old_logprob 及 tf.exp(logprob-old_logprob)、434–450 的 policy/ref_policy 角色。已從官方 HTTPS raw URL 重取並逐 byte 比對候選原碼。這是原作者2019實作，不當作當前 TensorFlow API 能在此環境執行的聲明。該實作會在minibatch重新whiten advantages；本文採樣前的基礎回饋／value不重算，教學程式的固定 A 則由本地實際核查佐證，不聲稱所有PPO變體皆固定同一種minibatch normalization。
- PyTorch2.14.1+cpu（git 5c4886908584029761b579af026dcfb627c84070）：親讀官方 _tensor.py:798–813 的detach語義與共享儲存警語、_torch_docs.py 的 torch.exp:4310–4334／torch.log:6289–6311、nn/functional.py:2293–2324 的log_softmax穩定性。HTTPS pinned版本重取後與已安裝原碼逐byte相等。CPU gradient、alias、clone及underflow/overflow實測均保存，沒有拿單次执行代替跨版本契約。
- 本地 tiny_perceptron/posttraining.py 全檔：ratio由exp(new_log−old_log.detach())計算，fixed_advantage亦detach；策略輸入四個結構特徵，輸出四張預寫卡片。scripts/course_experiments/posttraining.py 已親讀配置、split、_evaluate、reference複製、整個PPO loop、返回分母與timer邊界。收集位於no_grad，old selected clone，三個epoch不重新算分母；reference hash固定。只做3次2樣本的短SGD機制probe，沒有重訓正式排程。

形式實驗的既有原證據 docs/course-experiments/results/posttraining.json 為seed42、CPU、PyTorch2.14.1+cpu；代碼SHA與目前檔案一致。凍結權重SHA逐個核對，以目前原始_evaluate重新計算132訓練、15驗證、18測試題，全部165題各候選機率、RM分數、greedy動作／完整請求成功與保存報告一致（float32 abs≤1e−6）。完整逐題輸出在 frozen_all_rows.json，原始配置／split hash／checkpoints與分母在 execution.json。首批64個採樣动作×3epochs的192個ratio與reward−old value逐個重算，完整原欄位和重算比值在 formal_first_rollout.json。

有效token為0，PPO共120批×64=7680抽樣、重用23040次、policy與value各360次更新。既有PPO timer0.657001449秒包採樣、policy/critic更新、analytic KL與trace，排除SFT/RM/DPO、checkpoint保存及末尾凍結評估；experiment2.109786223秒與CLI2.122668108秒的起訖見 execution.json，未重測時間、未對GPU或普遍速度作結論。本節的20%、40%、0.6都是人工算例，沒有品質、吞吐或時間的實測主張；本實驗核查只佐證固定old/reference的程式機制。

13.12與明示13.11、13.4均沒有SVG；figure_consistency記not_applicable。對本節不需XML或渲染，不把沒有圖當作已渲染。

已核實條件：同一題與同一已採樣回答、正且有限的舊機率、同批歷史log保持不變、float32近似與顯示round區別、detach共享儲存、log保存不保證exp永不溢位、reference懲罰不是硬界。未發現需要正文修訂的矛盾。
