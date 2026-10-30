# 成熟模型有限研究規格

兩個研究都在自家十個正式工作完成、選定 weights／codec／T／tau／音型字典、獨立量測程式通過且 `final-opening.json` 已提交之後執行。成熟模型輸出不能用於自家訓練、調參、重挑 seed 或放寬門檻。Qwen3-Omni-30B-A3B 僅引用公開技術資料，不載入權重。

成熟入口在付費前從已完成的耐久帳讀出精確owned core ZIP，實際下載、逐檔核SHA，確認已提交manifest等於訓練時的原選擇，再重用正式開封的嚴格十job／五模組／checkpoint／calibration80／policy60／T／tau／map檢查。這一步只載入自製權重、沒有NN預測；不能以最小opening摘要或空bindings冒充前置完成。結果封包同時驗原始觀察與所有視角／固定家族的完整性，不能只有外層24／24／8清單長度；真正逐列失敗仍保留分母。

| 工作 | 固定版本與材料 | 固定形狀 | 全額保留上界 |
| --- | --- | --- | --- |
| Kev | `jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52`；base `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`；80 calibration、60 policy、60 final | L40S、4 CPU、48 GiB；直接 FP32 載入參考路徑；非 BF16 載入後再上轉 | US$4.30 |
| Qwen2.5-Omni | `Qwen/Qwen2.5-Omni-3B@f75b40e3da2003cdd6e1829b1f420ca70797c34e`；24 語音／文字配對、24 影音移時配對、8 短語音提示 | L4、4 CPU、48 GiB；以BF16請求載入完整模型，不量化；官方Token2Wav在聲音解碼轉FP32，另記各零件實際參數dtype | US$3.10 |

兩者沿用 Image `im-1b8GW96gjEVRiK1vg5Jqa0` 的 Python 3.12.10／Torch 2.8.0+cu128，不建 Image、Volume、Secret 或常駐服務。每次 Sandbox 最長 2,100 秒；其中 worker 1,800 秒包含安裝、下載與推論。單次 create／task allocation 等待最多 480 秒，清理查證共用 120 秒期限。沒有自動重派、延長等待或升級 GPU。

費率以官方 `Workspace.billing.rates()` 當次回應核對；GPU、CPU、記憶體、啟停、egress、7 天 artifact 保存與收尾全部先 CAS 預約。原有 US$60 gross round 與歷史 carry 原樣保留；不把費用預留當成實付，不假設退款或免費儲存。十個核心已在標準 CPU 完成後，另保留尚未嘗試的另一成熟研究與 US$2.00 final／發布餘額；同時防範官方費用延遲出帳。

Kev 程式固定 `jaredpalmer/kev@2074cc44256200bb37540103fa4572901faa10f3`，Apache-2.0。普通 Git 僅保存來源 URL、版本、授權及 SHA；執行所需原文放在外部輸入 ZIP，連同 LICENSE。adapter 的 head 與 config 另核已讀取的 SHA。關閉 fused、CUDA graphs，設定 head temperature=1 取得 raw logits，再分別呈現 raw、上游 T 與新 calibration T／policy。英文與中文只改寫同一 60 個公開 state，問題保持英文；不算成 120 個獨立情境，也不是不同語料來源或公平架構因果比較。dev 八家族比較未快取／快取、Q1→Q2→Q1 與實際 KV／recurrent／conv state 的 fingerprint。

Qwen 材料由已提交生成器在 Actions 中、載入成熟模型之前建立，保存 FFmpeg 版本、實際音訊／RGB bytes 與 SHA。這是自製合成英文問題與色彩世界；不宣稱人聲、自然度或獨立於小模型抽象情境的資料來源。固定 8 張 112×112 RGB、2 fps、完整 4 秒影音；processor 實測 grid `[4,8,8]`，input 上限 1,024 tokens。最多 104 次 Qwen generate：48 語音／文字、48 影音、8 語音輸出；thinker 上限 32／16 新 tokens、Talker 192；greedy、固定 Chelsie。沒有可播放 chunk 介面的 generate 延遲只稱完整生成時間。

固定版本不能只停在頂層revision參數：本輪實查Transformers4.57.6的Qwen載入器，附屬speaker檔會從已消耗的kwargs再取revision，可能落回main。因此先對Qwen與Whisper各取精確commit的完整必要snapshot，核LFS SHA256／Git blob及實際bytes，模型與processor只以local_files_only讀該目錄。輸出保留逐檔來源收據，不附上外部權重；並在Qwen載入後、八個speech嘗試後記Thinker／Talker／Token2Wav的參數numel/dtype，避免把載入請求當成全路徑精度。

八個輸出語音的內容另由 `openai/whisper-small@973afd24965f72e36ca33b3055d56a652f456b4d` 做文字內容 proxy，每次至多 32 tokens；不當作人聽、MOS 或自然度評分。所有錯誤保留原始列與完整分母。

輸入先固定 SHA，再保存在唯一、未發布的 owned draft Release，實際下載回核後才進 paid create。短 bootstrap 用 Sandbox filesystem 傳入至多 64 MiB／512 檔的輸入包；取回至多 128 MiB／展開 256 MiB 的自製結果與逐檔 manifest，先核 SHA 再停止精確 owned App／Sandbox。結果也保存到同一草稿，整個 Actions artifact 至多 256 MiB、control 16 MiB、保留 7 天。缺 task ID、身分不明、停止查證不完整或 artifact 超界，都不偽稱完成；預算不退回，也不盲目重跑。
