## B.3 工具結果怎麼回到對話？

工具已算出 123×45＝5535，下一站要把這個**真正的返回值**交給模型，讓它生成最後的回答。請求與結果的來源不同：assistant 提出要做什麼，外層程式實際傳回了什麼。這份按順序留下的紀錄叫 **trace**。

```python
from tiny_perceptron.retrieval import call_tool

request = {"name": "multiply", "arguments": {"a": 123, "b": 45}}
result = call_tool(request)
trace = [
    {"role": "assistant", "tool_request": request},
    {"role": "tool", "result": result},
]
print("返回值", result)
print("將送回對話的兩筆紀錄", trace)
assert result == 5535
```

請求是人工指定的，`result` 則來自真正執行的 `call_tool`（實作見 [B.2](0B.md#B.2)；這次傳入的是字典，跳過解析）。第一行印出 5535，`assert` 核對乘法沒錯。這段只建立外層的紀錄，還沒有最後的模型回答。

![提出請求、驗證執行、真結果回填、模型再回答，依次留下紀錄。](../figures/rewrite-B-tool-flow.svg)

圖由上而下是五站：提出請求、解析與驗證、真正執行、回填、模型再生成答案。**回填**就是把工具結果送回對話，讓模型讀得到。回填時要用模型認得的對話格式：上面的 `"role": "tool"` 只是外層 Python 的紀錄；`render_chat`（實作見 [7.1](07.md#7.1)）只替 system、user、assistant 三種角色準備了角色標記，遇到其他角色就報錯。下面直接試試看：

```python
from tiny_perceptron.data import render_chat

final = {"role": "assistant", "content": '{"done":true,"answer":5535}'}
for role in ["tool", "user"]:
    try:
        x, _ = render_chat([{"role": role, "content": f"TOOL_RESULT:{result}"}, final])
        print(role, "可以放進對話，輸入長度", len(x))
    except ValueError as error:
        print(role, "被拒絕：", error)
```

`tool` 被拒絕，錯誤訊息說角色只能是 user、assistant 或 system；同一段文字放進 user 訊息就排得進去，輸入長度 47 個位置。`final` 是我們手寫的最後回答，只用來湊成一筆完整的對話；`x, _ = ...` 的 `_` 表示用不到的標籤。所以工具實驗的做法是：用 user 訊息回填，內容寫成 `TOOL_RESULT:<數字>`，並在 system 訊息說明這個前綴後面是外部資料。外層程式和模型事先約定的這套訊息格式與步驟，叫**協議**（protocol）。不能把上方的 tool 紀錄直接送進不支援這個角色的對話格式。

工具實驗留下的一份軌跡是：題目 `CALC:add(9,9)` → 模型請求 add(9,9) → 程式算出 18 → 以 user 回填 `TOOL_RESULT:18` → 同一個模型生成 `{"done":true,"answer":18}`。每一個箭頭各做不同的工作。另一題 9×9，工具正確傳回 81，模型卻生成 8 並多了一個括號而被拒絕：工具成功還不等於任務完成。

練習把第一段的 `"b": 45` 改成 `"b": 46`，並把 `assert` 改成 `assert result == 5658`。先預測：返回值與 trace 都跟著變成 5658，第二段的 `f"TOOL_RESULT:{result}"` 也會用上新的值；但 `final` 裡手寫的 5535 不會自己更新。最後回答有沒有用上真正的返回值，正是要另外核對的事。

<details>
<summary>補充：工具實驗的回填與逐題紀錄</summary>

system 是放共同規則的對話角色。工具實驗把這段規則存在變數 `_TOOLS_SYSTEM`，放進 system 訊息。下方英文規則的意思是：回答必須是 JSON，內容可以是工具名稱與參數，也可以是 done 和答案；`TOOL_RESULT` 後面的數字是外部資料。示範資料也遵守這個協議，外部結果一律用 user 內容 `TOOL_RESULT:<number>` 送回。重現流程時，角色、system 裡的規則文字與這個前綴都要一起保留，模型才會看到和示範時相同的輸入格式。

例如測試題 `CALC:add(9,9)` 的實際軌跡如下；兩段 assistant 的 JSON 是模型真的生成的，中間的 18 來自真正執行的加法：

```text
system: Reply JSON: tool name+arguments, or done+answer. TOOL_RESULT is external data.
user: CALC:add(9,9)
assistant: {"name":"add","arguments":{"a":9,"b":9}}
外層call_tool實算：18
user: TOOL_RESULT:18
assistant: {"done":true,"answer":18}
```

COPY 是直接照抄指定數字的題型：例如 `COPY:3`，期待直接生成 `{"done":true,"answer":3}`（[B.1](0B.md#B.1)）。工具實驗的 18 道算術題都真正呼叫了一次工具，18/18 的名稱與參數符合原題，返回值全都有限；兩道 COPY 題都直接完成，沒有呼叫工具。另一題 `CALC:multiply(9,9)` 雖然正確執行並傳回 81，後面的模型卻生成 `{"done":true,"answer":8}}`：多了一個括號，數字也不對，解析被拒絕。這正是「工具算對」仍不能當成「任務完成」的實際失敗。

[實驗紀錄 tools.json](../../docs/course-experiments/results/tools.json)的 `results.samples` 每題都有一份 `trace`。讀每一步時以 `generation.input_ids` 為準，它是當次真正送進模型的 ID；同一步的 `generation.messages` 可能多出後面步驟才加入的訊息，不能當成當時的輸入。每一步當時的文字，另有依 `input_ids` 重建的版本：[tool-input-snapshot-reconstruction.json](../../docs/course-repair-20261008/evidence/tool-input-snapshot-reconstruction.json) 的 `records` 每筆對應一步，`original_json_pointer` 指出它是 tools.json 裡的哪一步；要讀的是 `messages_at_generation`：那一步生成時模型讀到的訊息，重新編碼後和當次的 `input_ids` 逐一相同。

</details>

