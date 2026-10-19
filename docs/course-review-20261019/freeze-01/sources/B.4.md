## B.4 任務完成後怎麼停止？

加法 2+3 已經算出 5，還需要一份 done 宣告與最後答案，才知道模型認為任務完成了。若只容許一個動作，可能已經算完卻來不及回答；若沒有完成訊號，外層程式也不能自己默認成功。

下面預先寫好兩個動作：add 與 done，當成測流程用的清單。工具實驗的模型不是一開始就生成整套計畫，而是每次只生成一個動作，外層執行並回填之後，再讓它生成下一個（[applications.py](../../scripts/course_experiments/applications.py) 的 `_tool_episode`）。課程函式庫的 `tool_loop` 依序處理清單裡的動作，最多處理 `max_steps` 個：

```python
from tiny_perceptron.retrieval import tool_loop

requests = [
    {"name": "add", "arguments": {"a": 2, "b": 3}},
    {"done": True, "answer": 5},
]
runs = [
    tool_loop(requests, max_steps=3),
    tool_loop(requests, max_steps=1),
    tool_loop(requests[:1], max_steps=3),
    tool_loop([{"name": "unknown", "arguments": {"a": 2, "b": 3}}]),
]
for run in runs:
    print(run["status"], "工具執行筆數", len(run["trace"]), "答案", run.get("answer"))
```

四次執行的狀態依次是 done、step_limit、needs_more_model_output、invalid_request，真正執行工具的次數是 1、1、1、0，只有第一次有答案 5。`run.get("answer")` 在字典沒有 answer 這個鍵時傳回 `None`（表示沒有值），不會報錯。這裡的 trace 每次成功的工具呼叫記一筆，請求與結果放在同一筆，和 B.3 那種「請求、結果」各一筆的對話紀錄不同。為什麼會有這四種結局？看原始碼：

```python
# 節錄自 tiny_perceptron/retrieval.py：tool_loop
def tool_loop(requests, max_steps=3):
    trace = []
    for step, request in enumerate(requests):
        if step >= max_steps:
            return {"status": "step_limit", "trace": trace}
        if request.get("done"):
            return {"status": "done", "answer": request.get("answer"), "trace": trace}
        try:
            result = call_tool(request)
            trace.append({"request": request, "result": result})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return {"status": "invalid_request", "error": str(error), "trace": trace}
    return {"status": "needs_more_model_output", "trace": trace}
```

`for step, request in enumerate(requests)` 逐一取出動作，`step` 從 0 起算（`enumerate` 見 [W.2](../first-steps.md#W.2)）。每個動作先檢查 `step >= max_steps`：已處理的動作數到了上限，就以 step_limit 結束。`request.get("done")` 在動作寫著 `"done": True` 時成立，傳回 done 與答案。其餘的動作交給 `call_tool`（[B.2](0B.md#B.2)）：成功就把請求與結果記進 trace；三道驗證有一道不通過，`call_tool` 就丟出 `ValueError`，由 `except` 接住，以 invalid_request 結束，`str(error)` 把錯誤訊息存成文字。清單全部處理完卻沒遇到 done，就落到最後一行：needs_more_model_output，表示還要模型再生成。

`except` 只接得住 `try:` 底下那兩行（呼叫 `call_tool`、寫進 trace）丟出的錯誤，而且要是括號裡列出的種類；其他錯誤會讓程式停下。`if request.get("done"):` 寫在 `try:` 之前，所以 `tool_loop` 收的每個動作都要是字典：若把 B.2 那種 JSON 文字直接放進清單，這一行就丟出 `AttributeError`（字串沒有 `get`），`except` 接不到，程式就此中斷，不會以 invalid_request 結束。括號裡的 `TypeError` 與 `json.JSONDecodeError`，要 `call_tool` 收到 JSON 文字這類不是字典的輸入才可能出現（B.2 的用法）；這裡的動作都是字典，碰不到它們。

下面這段要接在前兩段後面執行（見 [W.1](../first-steps.md#W.1)）：`requests` 來自第一段；執行過上一段節錄後，`tool_loop` 指的是節錄的定義，取代了第一段從函式庫匯入的同名函式，跳過上一段則仍是函式庫版。節錄的定義會用到同一個原始碼檔案裡的 `call_tool` 與檔案開頭 import 的 `json`，所以先載入：

```python
import json

from tiny_perceptron.retrieval import call_tool

for limit in [1, 2, 3]:
    print("max_steps", limit, tool_loop(requests, max_steps=limit)["status"])
```

上限 1 是 step_limit，2 與 3 都是 done；上限 1、3 的狀態和第一段用函式庫版算出的相同。`max_steps` 數的是處理過的動作，done 也算一個：add 用掉一步，done 還需要第二步，所以把上限從 1 改成 2 就能完成。清單用完卻沒有 done，表示還需要模型再生成；未知工具則在執行前就被拒絕。四種結局要分開保存，不能都叫「已回答」。

done 只是一個宣告：若寫成 `"answer": 6`，照樣回傳 done，答案卻是錯的。完整的評分要核對工具與參數是否符合原題、最後答案是否用了真正的返回值、內容是否正確；另外記下每次生成是否以 EOS 正常結束，以及真正呼叫了幾次工具。EOS（End of Sequence）是一次生成的結束標記，和 done 宣告整個任務完成不同（[7.9](07.md#7.9)）。

工具實驗的 20 道測試題裡，18 道算術題都真正執行了工具，2 道 COPY 題（照抄數字，[B.1](0B.md#B.1)）沒有呼叫工具；整題完成且答對的是 19/20。把上限改成一個動作後，18 道算術題全都在工具執行完之後停在 step_limit。這個對照教的是要預留回答的步數，不是計算器不會算。

練習把第一段 `requests` 第二項的 `"answer": 5` 改成 `6`，先預測：第一次執行仍是 done，答案卻變成 6，`tool_loop` 不會發現答錯；第三段三種上限的狀態都不變。

<details>
<summary>補充：工具實驗的 20 題、一步上限與生成次數</summary>

工具實驗每題最多生成三個動作，done 也算一個。下面核對同一批 20 題；「第一個動作解析成 JSON 物件」只代表得到一個 JSON 物件，不能單獨當成允許清單的驗證或任務成功。

| 核對項目 | 完整分母與結果 |
| --- | --- |
| 第一個動作解析成 JSON 物件 | 20/20題 |
| 合法 done、答案正確且完成要求的工具行為 | 19/20題 |
| 真正工具呼叫 | 18次，18/18名稱與參數符合原題 |

算術題要成功，還得真正執行符合原題的工具，而且最後答案等於真正的返回值；即使直接猜對數字，也不算照協議成功。COPY 題則要求沒有執行工具。再將同樣 18 道算術題的上限改成一個動作，模型仍各自真正執行一次，全部 18 題卻都成為 `step_limit`：沒有剩下的步數讓模型生成完成回答。這些是重新生成的 18 份軌跡，沒有拿前一次已完成的答案補進來。

正常流程裡，18 道算術題各生成請求與完成回答，共 36 次；兩道 COPY 題各生成一次，共 2 次。一步上限又讓 18 道算術題各生成一次，所以合計 36＋2＋18＝56 次實際生成。這 56 次全部以 EOS 結束，也沒有混入不該出現的特殊 token（例如角色標記，見 [B.2](0B.md#B.2) 補充）；但 56/56 的正常停止率不能代替 20 道任務的 19/20 成功率。

完整紀錄在[實驗紀錄 tools.json](../../docs/course-experiments/results/tools.json)：正常流程是 `results.samples`，一步上限是 `results.one_step_cap_samples`，成績是 `results.accuracy` 與 `results.status_counts`；每一步的輸入以 `generation.input_ids` 為準（見 [B.3](0B.md#B.3) 補充）。耗時也分開記：每次生成在 `generation.seconds`，整題從頭到尾在 `seconds`。這些實驗沒有一般自然語言選工具、任意程式執行或長程 agent 任務的成績。

</details>

