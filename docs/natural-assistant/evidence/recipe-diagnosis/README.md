這份證據包保存既有 train / validation 診斷、CPU meta 核對，以及原官方 model card / config / metadata 的 exact copies。結論見 RECOMMENDATION.md；逐檔來源、byte數與SHA256見 copy-index.json。這不是新模型版本、GPU驗證或新結果的選擇紀錄。

原始輸出仍在 `outputs/natural-extension/recipe-diagnosis/`。`scripts/` 保存當時在該 ignored 路徑執行的腳本原件副本；腳本用相對 parents 定位 repo，重跑時應放回其原路徑，且 tokenizer 分析需要原 session 的 pinned local processor cache。它們是可審閱的歷史執行來源，公開包不會自動執行它們。

`originals/qwen-4b/` 保存 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` 的已取得官方 metadata，不含 safetensors weights。`model.safetensors.index.json` 只是小型權重索引。此次整理沒有刷新官方 revision。

`originals/docci/` 保存 pinned 官方 README / website / license 與 `train_03125` 的單行 JSONL byte excerpt；其來源完整 JSONL 的 SHA256、行號與摘錄 SHA256 見 caption-boundary receipt。沒有複製完整 annotation corpus、圖片、重量或大型 TSV，也沒有審閱 test labels / outputs / images。

`history/RECOMMENDATION.before-caption-boundary-correction.md` 是修正前建議的原樣副本。`receipts/analysis-receipt.json` 是第一次分析的 immutable historical receipt，其舊 RECOMMENDATION hash 對應 history 中的版本；當前建議的SHA256與修正前後差異見 publication receipt / copy index。這保留更正過程，並避免把舊receipt當作修正後檔案的雜湊。

本次僅將「只取第一句」修正為「開頭片段，多數一句；句尾引號可能帶下一句」，核實例子是 `vision:docci/train_03125/caption`。Frozen manifest 仍為 `7604526c67da31a41940f16f9c87027c73cf2782c63522dbde8c7efbf015db91`。沒有改動凍結資料或模型程式，沒有推論或GPU。
