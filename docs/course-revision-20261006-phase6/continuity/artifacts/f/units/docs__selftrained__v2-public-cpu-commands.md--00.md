# V2 公開模型 CPU 操作命令

以下 7 組情境已用公開權重完成一次 CPU smoke，共 8 次 chat 呼叫與 1 次 history append；所有呼叫成功，回答與保存的 GPU validation 示範相同。命令保留當時實際 argv 的模型、訊息、預處理和生成參數，只把環境的 Python 路徑改為 `uv`，把檔案位置改為 repo-relative。原始執行記錄：`docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json`。

這些是挑選出的成功 validation 情境。它們可用來確認安裝、讀取權重與介面是否能運作，不代表未知問題的成功率。該 MoE 的最後一次 heldout test 中，完整工具往返為 **0/276**、語音回答為 **42/90**、語音後續對話為 **26/60**。這些限制不因示範成功而改變。

