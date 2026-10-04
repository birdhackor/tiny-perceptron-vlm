# 11.15 新版直式圖第二階段 factual 核驗

審閱者：`/root/natural_factual_mobile_11_15`；新 task，非本節 author 或 reader。

Root 通知第一階段 reader finalized pass 前，我不讀正文或新版圖；解除 gate 後開始自己的完整第二階段核驗。我實讀目前完整 11.15，以及明示必要前置 11.9、11.12 和前向連結 20.1。沒有讀 reader report、reader 練習解答或 reader render，沒有把 reader pass 作 factual 證據。

透明說明：解除 gate 後為了解既存 technical report 格式及保存歷史，我讀到了舊 `docs/technical-reviews/11.15.json`，包括旧 technical verdict，並已告知 root。本次沒有使用其判決、先前 owner 的 execution、notes、render 或練習解答作核驗證據。不能稱本 task 對舊 technical 結論完全盲目；可以確定下列核驗皆由本人重新完成。舊報告逐 byte 保存於 `docs/technical-reviews/history/natural-mobile-11.15-prior-bb293e22a5c290ea20075f0ac5514a4b95b526224a45d7f35b91449c50d7b77c.json`，SHA-256 為檔名中的完整 digest，舊 owner `/root/natural_factual_11_15` 的執行記錄仍屬歷史，不改署名。

## 原例與手算

讀 `natural_concepts.py:32–44`，確認輸入是單通道 batch tensor `(1,1,32,32)`，預設 CPU float32；第 15 欄整欄設 1，其餘為 0；`F.interpolate(...,(4,4),mode="area")` 得 `(1,1,4,4)`。回傳字典含六個實際用到的 key。原文自身沒有建中文字、辨識器、模型訓練或預訓練模型載入。

本人新執行 `section_facts.py` 的原文 fence。該次 worker exit 0、stderr 空、guard events 空，stdout：

```text
原圖與縮圖尺寸 [32, 32] [4, 4]
最亮的數字 1.0 0.125
至少半亮的數字個數 32 0
```

手算：32/4=8，每個輸出區域為 8×8=64 個值。含亮線的區域每列有一個 1、共八列，因此 8/64=1/8=0.125。有四個輸出區域含亮線，其他十二個為 0。輸入達到 0.5 的值有 32 個，輸出十六個值均不到 0.5，計數為 0。這不表示輸出全黑或無訊號：四個 0.125 仍非零。

本人另建獨立 CPU 檢查，手動把原 tensor reshape 為 `(1,1,4,8,4,8)` 並對兩個 bin 內軸平均，與 `area` 逐值精確相等；四行輸出均 `[0,0.125,0,0]`。float32 可以精確表示二進位分數 1/8；不需要用寬容差掩蓋差異。

## 已丟失位置與仍存在證據

另建第 14 欄全亮的圖，與第 15 欄全亮的原圖有 64 個不同輸入值，但 area 輸出完全相同。本人再枚舉亮欄 0–31，確認同一八欄區間內的八種細位置都得到同一縮圖；四個八欄區間仍產生不同輸出。故縮圖失去 bin 內細位置，不是失去所有位置或亮度資訊。

若 decoder 只收到同一縮圖、同一其他上下文，就沒有資訊能保證唯一選回這兩個不同原圖。增加 decoder 參數不改變相同輸入這一事實。先驗或句子可以猜中某個真值，卻不是確定復原保證。本節「未／末」是可能模糊的說明，不是已對這兩個實際中文字完成相同 tensor/OCR 失敗的測量。

## 本人實際讀取的原始來源

本次允許重用先前保存的原始来源檔。我複製原始 bytes 到自己的 prefix，記錄 reuse manifest，親自重讀原文；沒有聲稱自己重新下載它們。

- PyTorch 官方固定 commit `5c4886908584029761b579af026dcfb627c84070` 的 `functional.py`：實讀 `interpolate` docstring 5003–5041，N,C,H,W 接口和 area mode；5234–5237 的四維 area 分支派往 `adaptive_avg_pool2d`。本人比較該原始 bytes 與實際安裝模組，全檔一致。本次版本為 2.14.1+cpu。
- 同 commit `AdaptivePooling.h:31–37`：`start_index` 與 `end_index` 代入輸出大小 4、輸入大小 32，得到 `[8a,8(a+1))`；每個 bin 8 格，這個整數倍例沒有重疊。
- 同 commit CPU `AdaptiveAvgPoolKernel.cpp:17–69`：雙層 bin 內加總並 `sum/kh/kw`。本例 contiguous float32 以平均而非近鄰或 max 計算。
- LLaVA-UHD arXiv:2403.11703v1 §3.1：原生圖像分切、每窗視覺編碼，額外提供低解析度 overview；§3.2：更多高解析度視覺 tokens 增加工作量，但 compression 能降低後端 token 數；§3.3：逗號分列內窗、換行分行，保存相對位置；§4.1：每窗 64 tokens 時總量 64×(N+1)。此方法支持「全貌與細窗＋位置」的可用設計，沒有證明本課已實作该研究方法。
- Qwen2-VL arXiv:2409.12191v1 §2.1：可變解析度造成不同 token 數，packed length 控制記憶體，圖像 M-RoPE 用 height/width IDs；§2.2：圖文/OCR 與文字監督資料，而不是單亮線示例的模型能力；§3.3.1：單純放大並非總有提升，過度放大 OCR 小圖可能成為分布外輸入。只引用設計/限制，不把該論文 benchmark 結果當本課品質或成本實測。
- Qwen3-VL-2B-Instruct 官方模型卡固定 revision `89644892e4d85e24eaac8bacfd4f463576704203`：Key Enhancements 記已有圖文 pretraining，Quickstart 將模型及 AutoProcessor 都由既存權重 repository 載入；原始模型卡的宣傳能力描述不等於本次驗證。
- PEFT 官方 v0.17.1 adapter guide 的 Adapters 與 Low-Rank Adaptation：更新新增低秩矩陣，原始權重凍結，結果由底座與適配相加。正文「調整部分權重」在本路線含新增 LoRA 更新參數，不表示全部上游權重重新學習。

本人另外靜態實讀 `natural_assistant.py` MODEL_ID/REVISION、LORA_TARGETS、load_core 256–300、run_train 469–483：固定底座/處理器、凍結基底、只以 LoRA 參數建立 optimizer。未執行這個大模型路線，不能把靜態核對報成训练完成或品質改善。

數字 OCR 範圍另讀 `scripts/prepare_ocr.py` 的十個 GLYPHS、`draw_digits` 限定 ASCII 一至兩位數與 `range(100)`，以及 `modalities.py:833–856` 的數字 family 與 str(value) 答案。這足以確認該 16×16 路線的監督範圍，沒有重跑 GPU 實驗或修改資料。

## 新版直式圖本人渲染及查看

本版 SVG 是 720×1080，SHA-256 `3e4a4ddedd9558254a44c3a87f09a45f0f6abba45d565f6f67a910869a75901f`。本人用 Inkscape 1.4 各自 export-width 688、311，分別得 688×1032 與 311×467 PNG，再逐張 `view_image(detail="original")` 實際查看。標題、八欄、1/0、箭頭、1÷8、0.125、底部資訊損失及「沒有進行中文辨識」完整可讀、無裁字。

本人讀 SVG 幾何：左圖 x=120 至 600 寬 480；七條內格線每 60 分一欄，單亮欄寬 60；比率 60/480=1/8。箭頭由上方原區域向下方平均值，方向相符。輸出框放大供觀察，並不聲稱顯示為原 tensor 的真實像素大小。兩個 render 的輸出框內部取樣皆 RGB(32,32,32)，32/255 與 0.125 相差 0.000490196，比一個 8-bit 級距小；輸入深藍背景、格線與標籤是示意配色，不能當 literal RGB 平均實驗。

示圖只呈一個代表性 bin 的八欄與單亮欄，並非完整 32×32 到 4×4 截圖；它沒有把欄位排列或放大的格子偽裝成實際辨字結果。數值及「沒有進行中文辨識」和正文一致。模型的輸入解析度設計、模型參數容量與這張圖的顯示寬度各有不同用途；把教材圖變大不能被記成 OCR 能力提升。

## 本人完成正文練習

1. 放大回答的文字模型，增加可學參數與可表達/擬合的容量；更多容量仍需合適的資料與訓練，沒有保證。它不能在同一縮圖的多個可能原圖之間憑空建立唯一答案。
2. 在平均前保留較高解析度、用原圖细窗並保留位置，改善實際可提供的像素證據；已被模糊、遮擋或丟棄的筆畫不會因把低解析度圖插值放大就自動復原。全貌可提供跨窗關係，細窗可提供小字線索；成本取決於處理器、tokens 與 compression。
3. 提供中文字圖片及正確文字示範，教可見字形與應輸出文字的對應，也能教格式與看不清的回應；數字-only示範未提供這種中文監督。已有預訓練底座可能已知中文，本課新增示範也不能單獨保證未知字體、文件或遮擋情境都能讀對，仍需獨立測試。

三項對應容量、可用輸入資訊及學習對應。單一亮線的平均與 0.5 報告閾值，均不是中文字辨識器。本次 factual pass 只涵蓋正文、圖示、原例、透明數值與上述設計/限制；没有品質、速度或 GPU 記憶體實測。
