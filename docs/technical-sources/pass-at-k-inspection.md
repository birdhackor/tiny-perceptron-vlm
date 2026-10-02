# pass@k 與每題只抽 k 份的情況

這是作者於2026-10-02實際查閱原文後的補充筆記，不是逐節獨立技術審閱報告；審閱者仍須核對原文與當節主張。

實際閱讀 [Chen et al., Evaluating Large Language Models Trained on Code, arXiv:2107.03374v2](https://arxiv.org/pdf/2107.03374v2)，§2.1、式(1)、Figure 3。PDF首頁版本為2021-07-14 v2。該節先定義每題抽k份、至少一份通過的比例，再說這樣量的變異可能較大，提出以每題n≥k份、c份通過，計算 `1 - C(n-c,k)/C(n,k)` 的不偏估計量。原文說的是變異，不是把直接抽k份的指標一概判成有偏。

另實際讀取 [OpenAI原始實作](https://github.com/openai/human-eval/blob/6d43fb980f9fee3c892a914eda09951f772ad10d/human_eval/evaluation.py#L13) 的 `estimate_pass_at_k`；固定commit的檔案與初次下載內容逐byte一致。

透明推導：令n=k，分母C(k,k)=1。c=0時，分子同樣為1，所以結果0；c>0時，k-c<k，分子為0，所以結果1。因此標準公式在n=k時等於「本題至少一份通過」的指標。作者另以Python `math.comb` 逐一核對k=1/2/4/8與所有0≤c≤k，全部精確相等。

本輪算術報告在不同k各自真的抽一批候選，以 `oracle_coverage` 保存至少一份最後答案正確的題數／24。沒有另從較大的n樣本池估計多個k，也沒有跑HumanEval程式題benchmark。候選集合有正解仍不代表多數決或可部署的選擇器能選中；不同k使用不同隨機樣本，所以觀測比例也不必單調。不能寫成這個指標在所有情況下都與pass@k不同，或僅因n=k就有偏。
