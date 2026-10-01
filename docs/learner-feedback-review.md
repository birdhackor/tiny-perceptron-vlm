# 學習者提問、重現心得與教材補強

研究日期：2026-10-01（Asia/Taipei）。優先查閱 2024-10-01 至 2026-10-01 的公開案例。三個 subagents 分別研究 nanochat、MiniMind／MiniMind-V、CS336／LLMs-from-scratch 的實作者與讀者回饋。

## 1. 證據範圍

已讀 GitHub Issues／Discussions 的公開原文、留言與實際重現日記。一般搜尋引擎、部分個人網站、CSDN／知乎與 YouTube 受網路代理限制，未讀到的正文不作心得證據。

這是有目的的質性查閱，不能估計「多少學生」或難度排名，也不推定 GitHub 發文者都在校就讀。以下稱讀者、提問者或實作者。GPU 時數、生成樣本與修正效果是發文者的觀察，未在本環境重跑完整訓練。

明示為 AI 帳號的回答不作真人心得；由 AI 協助、但作者聲明本人 review／重現的案例，僅採其可追溯的觀察。歷史 bug 用來設計一般性反例，避免把暫時的版本問題變成模型原理。

## 2. 已安排的概念，需要講得更細

| 公開來源／月份 | 實際需求或困難 | 對教學的調整 |
| --- | --- | --- |
| [nanochat Discussion #365](https://github.com/karpathy/nanochat/discussions/365)，2025-12；[MiniMind #584](https://github.com/jingyaogong/minimind/issues/584)，2025-12 | 前者找逐步讀 repo 的指南；後者留言要求訓練步驟對應的程式檔與啟動方式。 | 每節指定少量核心程式、入口、資產和預期輸出。先單例，再 batch；提供固定版本的練習副本與文字 diff。 |
| [nanochat Discussion #856](https://github.com/karpathy/nanochat/discussions/856)，2026-09 | 成功預訓練四層模型，回頭讀碼仍卡在 `backward()` 與 optimizer。 | 先微小擾動估梯度，再手算小鏈條、對照 autograd，最後看清零、backward、step 的順序。 |
| [CS336 #29](https://github.com/stanford-cs336/assignment1-basics/issues/29)，2025-09；[#72](https://github.com/stanford-cs336/assignment1-basics/issues/72)，2026-01 | Q／K 與 FFN 的記號、shape 和程式方向不一致；維護者確認教材問題。 | 軸名稱與 row-vector convention 全程一致。用非方陣展示 `Linear.weight=(out,in)`、`x @ W.T`，再加 batch/head。 |
| [MiniMind #476](https://github.com/jingyaogong/minimind/issues/476)，2025-08 | 不懂 assistant-only loss mask 與 attention mask 的差別。 | 同一對話並排 causal、padding attention、supervision 三種 mask；改 user 內容看回答 logits，檢查忽略 loss 的位置仍可經上下文影響梯度。 |
| [MiniMind #633](https://github.com/jingyaogong/minimind/issues/633)，2026-01 | `start+1` 是否漏掉 assistant 第一個 token？維護者用逐位置 `X → Y → mask` 表解釋。 | 6–12 token 表列 ID、角色、輸入、下一位置 label、loss 權重；手算一次 CE。Shift 只在定好的接口做一次。 |
| [MiniMind-V #98](https://github.com/jingyaogong/minimind-v/issues/98)，2025-11 至 12 | 困惑 projector-only pretrain 的 loss 平台、基模類型與階段目的。 | 每階段列已有能力、可更新參數、梯度、optimizer 收錄與權重變化；以圖片敏感度、圖文答案和文字保留評估。此 issue 是歷史 CLIP 版本，不能套用到後來 SigLIP2 配置。 |
| [MiniMind-V #8](https://github.com/jingyaogong/minimind-v/issues/8)，2024-10 | 問哪個 CLIP encoder 接入 LLM、token 數、patch size 是否能直接改。 | 畫 encoder→projector→圖片槽的 shape；用前處理前後圖片區分改輸入、改架構和使用相容權重。 |
| [LLMs-from-scratch #1079](https://github.com/rasbt/LLMs-from-scratch/issues/1079)，2026-09 | LoRA 的 alpha 與 alpha/rank convention 不一致；相近 loss 的短句生成也不同。 | 用一層 Linear 看 rank、alpha 和實際縮放；記錄 convention。生成除錯列 top-2 與首次分歧，避免要求逐字重現參考答案。 |
| [MiniMind #515](https://github.com/jingyaogong/minimind/issues/515)，2025-10 | 把身份 LoRA 的低 loss 閾值套到醫療 LoRA，追 loss 卻見原能力下降。 | 比較小型身份記憶與多樣新任務；保留 validation／舊能力，不把某個 loss、LR 或 epoch 數當通用成功標準。 |

## 3. 值得補的小知識與失敗實驗

| 公開來源／月份 | 缺口 | 本課安排 |
| --- | --- | --- |
| [CS336 #21](https://github.com/stanford-cs336/assignment1-basics/issues/21)，2025-08 | 單 token 可能只有部分 UTF-8 codepoint，逐 token decode 出現替代字元。 | 中文字／emoji→bytes→token→完整串接 decode 的小表；區分字元、byte 與 token。 |
| [nanochat Discussion #802](https://github.com/karpathy/nanochat/discussions/802)，2026-07 | 混淆 subword、句子邊界、訓練 window 與長距離能力。 | 同一 token 列表切短窗口，畫 shifted targets；對話另列 history＋prompt＋response 預算。 |
| [nanochat #590](https://github.com/karpathy/nanochat/issues/590)，2026-03 起 | SFT NaN 最早被猜為小 batch 問題；後續重現指出短 context／packing 可產生永遠全 ignored targets。有人加零 loss 防護後仍完全沒學。 | 正常、答案被截光、答案/EOS 被截一半三種反例；每個 batch 計有效 assistant targets，定位原始樣本。全 ignored 的 mean loss 不作成功分數。 |
| [LLMs-from-scratch #566](https://github.com/rasbt/LLMs-from-scratch/issues/566)，2025-03 | 最後位置落在 PAD，分類輸出隨 padding 長度變化。 | 兩筆長短不同的序列，比較 loss mask、padding attention 和最後有效輸出位置；再移用到 batched generation。 |
| [CS336 實作日記](https://github.com/donglinkang2021/cs336-assignment1-basics/blob/c7bd4a373c1c56fc6cb02cc3f848c13d02a2b825/docs/CHANGELOG.md#010---20251012)，2025-10 | 作者坦承固定小 LR 試錯很久；Muon 比較混入 LR 差異。 | 三種 LR 的短曲線先建立可學基準；固定同一 LR 的概念對照與各方法合理調參的比較分開。 |
| [MiniMind #732](https://github.com/jingyaogong/minimind/issues/732)，2026-04 | 直接要求原始資料如何處理和配比。正文沒有更多說明。 | 展示原始 record、保留/淘汰理由和配方；兩任務 90:10／50:50，記錄實際筆數、有效 tokens 和分項結果。 |
| [nanochat Discussion #846](https://github.com/karpathy/nanochat/discussions/846)，2026-09；[#819](https://github.com/karpathy/nanochat/discussions/819)，2026-08 | baseline 分數波動；重現者提醒不同 commit／配置的略好分數不足以支持方法更好。 | 三次 tiny seed、固定 protocol、保留原始樣本；畫分數範圍，標出本次只支持的結論。 |
| [nanochat #860](https://github.com/karpathy/nanochat/issues/860)，2026-09 | 提問者猜測 GSM8K 分差可能含答案格式因素；仍未拆解證實。 | 相同數值寫成字串／JSON／自然句，分別量內容正確與格式符合；查看 parser 與原始生成。 |
| [MiniMind-V #36](https://github.com/jingyaogong/minimind-v/issues/36)，2025-02 | 問 OCR 要新程式或新標註資料。 | 圖文支線：手製合成數字→多位短字串；逐字與整串評估，連到解析度／裁切。真實中文 OCR 另列資料與資源需求。 |

## 4. 實際重現方法可以借用，結果須保留條件

- [nanochat Discussion #677](https://github.com/karpathy/nanochat/discussions/677)，2026-03：第一次完整使用流程的帶讀，逐步列產物與 Success 判準。CPU smoke 的亂碼是可預期結果；用它教「能執行」與「模型任務達標」的不同驗收。
- [nanochat Discussion #710](https://github.com/karpathy/nanochat/discussions/710)，2026-04：homelab 使用兩台 DGX Spark，分享時數、問題與 fork。不能把其時數套成本環境 CPU。
- [nanochat Discussion #819](https://github.com/karpathy/nanochat/discussions/819)，2026-08：單 RTX 5090 的長時間實跑，報告 backend、batch、logging、peak memory；本人對配置不匹配的比較保持保留。借用成本分欄與對照方式。
- [MiniMind-deep-dive 的評估紀錄](https://github.com/Enping-Hu/minimind-deep-dive/blob/18a77db608480d88cd77af91fa1a8705f516e56f/chapters/10-experiments/03-eval-conclusions-sft-vs-rl.md)，2026-06 至 07：固定八個 prompt 比較階段 checkpoint，發現長度／格式改善仍有答錯，MoE 單條 decode 也可能較慢。其「SFT 不學事實」「必然是 reward hacking」等因果斷言超出小樣本，未採用。

其他值得借用的呈現方式：[MiniMind-in-Depth](https://github.com/hans0809/MiniMind-in-Depth) 的逐行／shape 解說，以及 [learn-minimind-wrz](https://github.com/1209244478/learn-minimind-wrz) 的 CPU 獨立小節與原碼映射。README 的宣稱未經本課執行驗證，不直接視為效能保證。

## 5. 操作摩擦另放實作指南

| 來源 | 問題 | 採用方式 |
| --- | --- | --- |
| [nanochat #542](https://github.com/karpathy/nanochat/issues/542)，2026-02 | 自動載入了另一個 model tag，誤以為目前模型沒學。 | Notebook 開頭明列 checkpoint 路徑／ID、step、tokenizer 和架構；載入前核對相容性。 |
| [LLMs-from-scratch #1015](https://github.com/rasbt/LLMs-from-scratch/issues/1015)，2026-04 | 改 Notebook 做探索後，難以合併上游 JSON 更新。 | 固定教材版本、練習副本和容易 diff 的文字來源；不用為此立即加同步工具依賴。 |
| [CS336 #89](https://github.com/stanford-cs336/assignment1-basics/issues/89)，2026-06 | Windows 被 UNIX-only 測試依賴卡住，平台需求不清楚。 | Quickstart 明列已測平台和最短 smoke；系統選修另列環境需求。 |
| [MiniMind-V #114](https://github.com/jingyaogong/minimind-v/issues/114)，2026-04 | batch=1／多圖有 shape bug，作者確認修正。 | 用 batch=1、sequence=1 的邊界形狀練習；不宣稱所有多圖原理都難懂。 |

## 6. 融入規劃的方式

優先補強兩條可觀察流程，各步分節：

1. **一筆資料：**raw record → 格式／token → X/Y → masks → batch → logits／loss。
2. **一次更新：**數值擾動 → chain rule → autograd → 清零／累積 → step → 梯度與權重變化。

每節提供一個成功例、一個只破壞一項設定的失敗例，以及讀者能用來定位的觀察。教學內容與原碼映射、shape、token 表、有效監督計數、已測配置都列為驗收項。

本輪新知識以梯度生命週期、UTF-8／token、padding 輸出位置、短 LR 搜尋、資料配方、context budget 和合成 OCR 為主。評估波動、LoRA 縮放、凍結參數與模型資產主要是既有單元的深化。

兩個額外辨識題可保留為選修：

- [nanochat Discussion #420](https://github.com/karpathy/nanochat/discussions/420) 有讀者問 2B 8-bit 是否等於 1B 16-bit；沿用第 17.1 節，分開比較純權重 bytes、參數量與實際運算成本。
- [LLMs-from-scratch #848](https://github.com/rasbt/LLMs-from-scratch/issues/848) 討論 response-only／full-sequence SFT；作為標註目標的比較，不當成任何一種永遠最佳的證據。

具體小節與跳讀路線見 [教學大綱](curriculum.md)；近期官方課程對照見 [公開課研究](public-course-review-2025-2026.md)。
