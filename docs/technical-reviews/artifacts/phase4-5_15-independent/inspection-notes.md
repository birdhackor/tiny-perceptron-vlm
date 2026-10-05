# 5.15 獨立來源查證與邊界

本輪身分 `/root/phase4_factual_coordinator/factual_5_15`，只判 5.15。親讀目前 UTF-8 原節、fence，必要前文 3.5（標準差）及 5.14（同起點比較）。先親讀 factual-reviewer-instructions.md、完整 checker、完整 section_facts.py 與 review-protocol.md；未讀舊技術或讀者報告。正文、圖與產品碼均未修改。

使用 cloud-environment-onboarding:setup 技能確認既有 `.venv` 可執行相稱 CPU 工作；沒有安裝、環境設定或 draft 修改需求。Python 3.13.5、PyTorch 2.14.1+cpu、CUDA build None、CUDA available False。官方來源使用固定 PyTorch v2.8.0 原碼及 2.8 文件；這與安裝版不同，故同時親讀安裝版契約並執行行為，沒有宣稱版本相同。

## 親讀權威原件

- 原作者 arXiv `2103.03098v1`，2021-03-01，Accounting for Variance in Machine Learning Benchmarks；PDF SHA 與 HTTPS URL 在 extra-acquisition.json。親读實體 PDF 第 1–5 頁相關論述、第 7–10 頁 §4–§6 相關段落（非附錄全文）。第 1–2 頁 §1/§2.1 說明只量初始化未覆蓋其他波動；第 3–5 頁 §2.2/§3.1，有限測試集與相關錯誤改變有效分母、重複數影響估計；第 7–8 頁 §4.1，用配對風險的分布而非單平均比較；第 9–10 頁 §5/§6，樣本、資料集、多次比較影響結論。支持本文的定性限制；沒有移植論文測試閾值、錯誤率或 GPU 成績到本文。
- 先嘗試錯誤的 PMLR 139 文章位置取得 404；已保留 acquisition.json 的真實失敗，再由作者 arXiv 原始 PDF 補齊。沒有拿搜尋摘要作證据。
- PyTorch v2.8.0 `torch/random.py` 10–60 行：get/set_rng_state 操作 default generator；manual_seed 的種子是生成器的起始狀態設定，並無品質順序語義。
- PyTorch v2.8.0 `docs/source/notes/randomness.rst` 全文：首段跨版本、裝置、平台不保證重現；PyTorch random number generator 小節中重設與連續呼叫的区别；Python/NumPy 各有來源；DataLoader 小節使用獨立 generator/worker 初始化。本文程式是相同版本、CPU、同架構重建；不把此結果延伸到任意完整 GPU 訓練。
- PyTorch v2.8.0 `torch/nn/modules/sparse.py` 15–44、135–190 行：ID 查表、可學權重形狀、預設 N(0,1)、normal_ reset；安裝版 Embedding 同樣查表與初始化。TinyLM 原碼 14–28、53–86 行實際使用 vocab_size=264、width=8、nn.Embedding，沒有自定 generator；attention 31–46 與 modern 35–43 行都是預設線性層初始化。安裝版 Linear/LayerNorm reset 原碼已親讀，前者抽隨機值、後者固定 ones/zeros。
- PyTorch v2.8.0 `torch/nn/init.py` 75–82、240–264 行：normal_ 預設 mean=0、std=1、generator=None，no_grad 寫入值。安裝版同契約；manual_seed 安裝版改為呼叫 `_manual_seed_impl`，親讀後者確認最後呼叫 default_generator.manual_seed。
- PyTorch 2.8 `torch.std` 原始 API 頁全文主區域：dim=None 降所有軸、correction=1。2.8 `torch.equal`：相同大小與元素，無近似容忍（NaN 例外與本節初值無涉）。2.8 `Tensor.item`：單元素 Tensor 轉 Python 數字。安裝版三項 __doc__ 也另保存。
- PyTorch 2.8 `torch.Generator` 主區域前段/get_state/manual_seed，`torch.randperm` 主區域：generator 管理自身狀態，randperm 接受 generator；用來核對正文「必要時分開模型與資料亂數」的可實作範圍。

## 算例、程式與結論

原 fence 經 section_facts.py 原樣執行、退出 0、沒有 guard events：std=1.035154938697815、1.0004743337631226、1.0190640687942505；same seed True。這些是全 264×8=2112 個初始化值的樣本標準差，不是三個成績、模型品質、梯度或更新。

自己保存並真的执行精確練習變化：只移除原碼最後一次 `torch.manual_seed(1)`，True 變 False。額外 CPU probe 親量建構前後 RNG state 都前進；再次重設可重建全部相同參數。ID 0、1、263 查到對應權重排；權重 requires_grad=True，grad=None。本次沒有 backward 或 optimizer。

std() 預設分母為 2111，自行以 double 重算 sqrt(sum((x-mean)^2)/(2112-1))=1.0351549596293472；與 float32 std 相差 2.09e-8，容忍 1e-6。接近 1 的理據是 N(0,1) 初始分布；三個有限抽樣不保證恰為 1。

相同 seed=1 分別初始化 width=8 與 width=16，再抽 20-ID permutation：兩個順序不同；事先獨立設 data Generator seed=17 後，兩種架構得到相同順序。這是共用流錯開的存在例，不宣稱任何兩架構一定不同或實際 train loader 已分流。

以 Decimal 自算手工 A/B 三組數字：B-A=+0.01/-0.01/+0.04；B 最大值0.73，A最大值0.72，但第二組較差；A 平均 0.703333…、B平均0.716666…，平均差0.013333…；A範圍[0.69,0.72]、B[0.71,0.73]。分母是每方法三個假設數與三對差，2對正、1對負。原文明确「假設實際測得」；没有 metric/dataset/token/task 具體成績來源，故不能標成新 empirical 結果，亦沒有顯著性檢定或普遍 B 較好的推論。

「家族」按題目分組/涵蓋的資料群理解：不能把高度相關題目當成額外獨立證據；原文沒有指定一套 group 數值或统计算法。原論文 §2.2（相關錯誤）、§6（跨資料集）支持這個定性邊界，並不证明某個教材任務的有效樣本數。

無既有實測 JSON 引用，未重訓或產生新模型成績。沒有圖片/SVG 引用，未宣稱 render 或頁面檢查。本節數字與狀態變化由明確文字和程式充分呈現，不需要自行想像空間/素材；figure_consistency 為 not_applicable。

目前未發現影響理解的錯誤或不確定主張，建議 pass。證據範圍是這組算例、目前 TinyLM 初始化及有界 RNG 變化；成熟統計限制由親讀原論文支持。任何版本變動仍需重新核對本人的原稿/圖/證據指紋。
