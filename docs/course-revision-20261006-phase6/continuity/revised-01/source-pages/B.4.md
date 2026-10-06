## B.4 任務完成後怎麼停止？

加法2+3已回5，還需一份done宣告與最後答案，才知道模型認為任務完成。若只容許一個動作，可能已經算完卻來不及再答；若沒有完成訊號，不能由外層默認成功。

下面預先給兩份動作：add與done。這是測流程的清單，不是模型一開始就生成了整套計劃。

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

四份run的status依次是done、step_limit、needs_more_model_output、invalid_request；真工具次數1、1、1、0。trace在此每次運算記一筆，不是上節請求／結果兩筆對話。只有第一份有答案5。

`max_steps`計處理的動作，done也算一個，改上限1為2讓第二份能完成，沒有逼它多用工具。清單耗盡但沒有done，需要後續模型輸出；未知工具則在執行前拒絕。四種結局應保留，不把都叫「已回答」。

done只是宣告，若answer=6，仍會回done卻答錯。真模型應每次生成一個動作，執行與回填後再生成下一個。完整評分核對工具是否符合原題、真返回值與最後內容，再另查每次生成是否以 EOS 正常結束、呼叫數與等待。EOS（End of Sequence）是一次生成的結束標記，與 done 宣告整個任務完成不同，詳見[7.9](07.md#7.9)。

舊20題固定協議裏，18題算術真執行，2題[COPY照抄數字](0B.md#B.1)不呼叫；整題19/20完成。一動作上限則讓18題全在工具已執行後step_limit。這個對照教的是預留回答步數，不是計算器不會算。

<details>
<summary>補充：舊20題任務、一步上限與生成分母</summary>

正式組每題最多生成三個動作，done也算一個。下面核對同一20題；「首動作解析」只代表得到JSON object，不能單獨當成白名單驗證或任務成功。

| 核對項目 | 完整分母與結果 |
| --- | --- |
| 首動作解析成JSON object | 20/20題 |
| 合法done、答案正確且完成要求的工具行為 | 19/20題 |
| 真正工具呼叫 | 18次，18/18名稱與參數符合原題 |

算術題的成功另外要求真正執行對應原題的工具，且最後答案符合真實返回值；即使直接猜對數字，也不能算這份工具協議成功。COPY題則要求沒有執行工具。再將相同18道算術題的上限改成一個動作，模型仍各自真正執行一次，全部18題卻成為`step_limit`：沒有預算讓模型生成完成回答。這些是重新生成的18份軌跡，沒有拿前一次已完成的答案補進來。

正常流程裡18道算術題各生成請求與完成回答，共36次；兩道COPY題各生成一次，共2次。一步上限又讓18道算術題各生成一次，所以合計36＋2＋18＝56次實際生成。這56次全部有EOS、沒有混入非法控制token（如[B.2](0B.md#B.2)所述的專用角色標記）；56/56停止率仍不能替代20道正常任務的19/20成功率。完整紀錄見[實測報告](../../docs/course-experiments/results/tools.json)；各次生成與整個任務的等待時間應分開記錄。本輪沒有一般自然語言選工具、任意程式執行或長程agent任務的成績。

</details>

