# v4 validation 的 blind grading 與選版

這份是作者執行說明。`scripts/score_natural_v4_validation.py` 只讀取已完成的 validation artifacts，不呼叫模型，不改 gold，也不執行 test。

下載真實 validation 的 `result.json`、三個 `generations-*.json` 與 `transcripts.json` 到同一資料夾後，匯出 blind packet：

```bash
.venv-natural/bin/python scripts/score_natural_v4_validation.py export \
  --validation-dir outputs/natural-v4/validation/validation-run-id \
  --blind-dir outputs/natural-v4/validation-review/validation-run-id
```

預設使用 `docs/natural-assistant/v4/manifest.json`、`validation-protocol.json` 及 `outputs/natural-v4/data`；可分別以 `--manifest`、`--protocol`、`--data-root` 指定本機路徑。

把 `blind-packet.json`、`grades-template.json` 與實際來源圖片交給 fresh reviewer。**不要提供 `private-map.json` 或原始 candidate filenames。** packet 只有隨機 A/B/C、完整原始回答、正式問題／上下文／rubric／參考示例、來源圖片位置及 decoder-complete 狀態。别名映射使用 private nonce 的 hash commitment 固定，評分後可連同 map 一起 commit，供選版追查。

目前需要 303 個人工 semantic grade（每候選 84 photo、9 text chat、4 typed-reference voice、4 actual-ASR voice），其餘 31 個 exact 題由程式按既有 runtime normalization 重算。每個人工 grade 需填 Boolean `passed`、非空 `reason`；完成的 photo 回答必須實際看圖，填 `source_image_inspected: true`。若 decoder 未確認 EOS，答案本就判錯、adapter 不 eligible；此時不需要為 semantic grading 再看圖，但仍要記錄 grade 與理由，不能縮小分母。

reviewer 在模板頂層填 `grader`，保存為 `grades.json`。其 `blind_packet_sha256`、case ID 與 alias 不得改動。使用：

```bash
.venv-natural/bin/python scripts/score_natural_v4_validation.py score \
  --validation-dir outputs/natural-v4/validation/validation-run-id \
  --blind-dir outputs/natural-v4/validation-review/validation-run-id \
  --grades outputs/natural-v4/validation-review/validation-run-id/grades.json \
  --output outputs/natural-v4/validation-selection/validation-run-id
```

輸出 `scores.json` 的完整 counts、EOS／nonregression gates 與精確整數 macro；`selection.json` 是現有 Modal pre-test gate 可接受的 signature，綁定 actual validation result、manifest、protocol、人工 grade、blind packet、scoring SHA，以及選中 archive 的 train-run／checkpoint／safetensors SHA。先審查並 commit 決策，再執行一次新 test。

輸入驗證會拒絕不同 manifest／ASR pin、漏或重複 generation／grade、未知題目、候選 ASR user 不同、候選數不完整、別名映射被改動，或舊 packet 對上新的 artifacts。`result.variant.completed` 只算行數，程式另逐行檢查 raw EOS，且接受恰好第 384 個 token 是 EOS。primary 直接比較整數 `/75600`；真 tie 保留 base 或較早 archive，`base minus 1` 的 minimum 將下界限制為 0。

base fallback 仍可能有判錯或 decoder 未完成的題目；選 base 不代表所有能力測試通過。現有 Modal guard 驗證的是已完成 validation 與權重綁定，並不自行重算人工 rubric；本 scorer 的 grade／packet／scores 證據要連同 selection 一起保存。
