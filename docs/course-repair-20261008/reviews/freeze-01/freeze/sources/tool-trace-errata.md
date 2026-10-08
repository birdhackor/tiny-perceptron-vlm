# 工具實驗逐步輸入紀錄勘誤

本勘誤適用於 [B.3](../../course/chapters/0B.md#B.3)、[B.4](../../course/chapters/0B.md#B.4) 及 [T.11](../../course/training.md#T.11) 所引用的 [tools.json](../course-experiments/results/tools.json)。原檔保留，SHA-256 為 `3befd7fcb9a1a48d850761ab6d0917ac42388dc3bbfe9f011f0b98c3752d7140`。

每步的 `generation.input_ids` 是實際送入模型的編號；`generation.messages` 原本應保存同一次輸入的文字對照。舊程式卻保存了外層仍會繼續加入訊息的同一份列表。因此，早先步驟的文字欄位可能已包含後來生成的工具請求與工具回傳，不能拿它當作當時模型已看見的內容。

核對範圍是 20 個正常任務與 18 個「最多一步」任務，共 56 次生成；其中 36 次的 `messages` 與當時輸入不同。這是紀錄快照問題。從每題保存的起始系統訊息與使用者題目開始，只加入先前步驟實際生成的請求和已執行的有限工具結果，可以逐步重建輸入；全部 56 次重建都與原有 `input_ids` 逐個編號一致，編號總數也相同。

[輸入快照重建補件](evidence/tool-input-snapshot-reconstruction.json) 用 `original_json_pointer` 定位原檔的生成步驟，`messages_at_generation` 保存重建的當時文字，並保留逐 ID 核對結果。這份補件是從舊紀錄重建的文字對照，並非重新執行模型的觀察。

原檔中的生成答案、工具執行結果、真正輸入編號和正常任務的 19／20 計分不因本次重建而改動；也沒有藉此增添新的模型能力證據。日後的 `_sample` 會保存訊息的深層副本，使後續回填不再更動早先的輸入快照。

可在專案根目錄執行以下離線核對。程式先核原檔指紋，再核每個步驟；不訓練、不呼叫模型、不覆寫原檔，既存補件不同時也會停止。

```bash
python docs/course-repair-20261008/evidence/reconstruct_tool_snapshots.py
```
