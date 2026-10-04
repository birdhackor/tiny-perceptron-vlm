# 既有 ASR validation 比較的正式來源核查

`source-bound-verification.json` 是原作者對既有 evidence 的重核查，不能替代後續獨立 correctness reviewer，也不是新的模型評測。只檢查已用於 ASR 選版的 12 筆 FLEURS validation 與 4 筆 AISHELL validation；沒有新的語音／LM 推論，沒有讀取 test transcript 作為評測 reference，沒有載入權重。

核查程式用兩個固定 revision 的本機 Whisper tokenizer，把全部 32 份已保存 raw token IDs 重解碼，結果均等於原始 transcript。也核對原始 decoder prompt 是中文、transcribe、no timestamps；真實 EOS、token 數與未達 128 tokens 上限的記錄一致。

兩模型保存的資料 ID 順序、原始錄音 SHA／bytes／sample rate／channels／duration、資料來源 member、原始 reference 字串 SHA、greedy decoding 設定、CPU float32、5 threads／interop 1 均相同。參數量 241,734,912（small）與 808,878,080（turbo）來自先前實際載入後 `numel` 的 stdout，本次只核對 stdout 與 summary 一致，未重新載入權重。

另以獨立的二維 DP 重算原始 Unicode-codepoint CER，以及 NFKC 加移除 Unicode whitespace 的 CER。標點與繁簡字形維持原始資料政策，沒有先改 reference 再評分。32 筆 numerator、denominator 與各來源／總體聚合值都和保存的 `comparison.json` 一致：small 原始 138/524、normalized 117/510；turbo 原始 68/524、normalized 50/510。

root 已根據這一組 validation 固定選 turbo，之後 base 與 LoRA 共用同一 ASR 選項。這些錄音是公開人工朗讀的陳述與問句，這次結果没有證明自發對話、任意說話者、聊天回答品質或 test 泛化能力。`comparison.json`、既有 protocol／records 保持原字節；這份核查新增的程式／輸出 SHA 存在 `source-bound-verification-copy-receipt.json`。
