## 19.7 計算器算對，為什麼助理還可能答錯？

你請同學向計算器確認1加2，他可能按錯數字，也可能讀到3卻在回覆時寫成4。工具使用正是這種兩段工作：模型先寫請求，程式執行，再由同一模型讀結果與回答。若最後直接把程式算出的3當成模型答案，會把「會讀回結果」這項能力藏起來。

前置是[工具請求](0B.md#B.1)、[驗證與執行](0B.md#B.2)、[結果回填](0B.md#B.3)、[停止原因](0B.md#B.4)與[本成品的動作／EOS](19.md#19.1)。B章用JSON請求單；這個小成品將同樣欄位縮成`TOOL:calculator:1+2`，依序是動作、工具名與參數。byte是儲存資料的位元組；這段英文字母、數字與符號各佔一個byte，在本小模型裡各佔一個需要生成的位置。縮短請求就減少需要生成的位置。本成品只開放一個calculator工具，接受有限範圍的非負整數加法。allowlist是明確允許的工具名清單；模型寫出未列入的名字時，程式拒絕執行，而不是隨意把文字當成Python程式碼。

![模型先提出請求，程式驗證與計算，再把實際結果交回同一模型生成最後回答](../figures/capstone_tool_loop.svg)

```python
from tiny_perceptron.capstone import calculator_runtime, parse_action

trace = {"raw": "TOOL:calculator:1+2", "eos": True}
action = parse_action(trace)
print("解析後的請求", action)
for available in (True, False):
    result = calculator_runtime(action, available=available)
    print("計算器可用", available, "實際執行結果", result)
print("還沒有模型讀回結果，也沒有最終模型答案")
```

`trace`是本例人工填寫的紀錄字典，`raw`保存手寫請求文字，`eos=True`也是人工宣告「此例已正常結束」；不是模型真的生成EOS的證據。`parse_action`把它解析成`{"status": "tool", "name": "calculator", "a": 1, "b": 2}`：請求種類是tool、工具名是calculator，兩個數字已轉成整數。這份`action`才是`calculator_runtime`收到的輸入。

請求是人工指定的，但工具在這裡真的執行了。可用時應得到`status=ok`、`result="3"`；關閉時應得到計算器不可用的錯誤。它不會因為加法很簡單就繞過開關，也不會把錯誤當成功結果。最後一行故意提醒目前的界限：parse與工具成功，只是整個流程的中間站；後面的實測才有模型生成的請求與EOS。

正式`run_assistant`先保留模型的`action_trace`，再記錄`runtime`；若執行成功，把原題與真結果組成「原題：1+2。計算器回報：3。請回答。」這份新的user前文，呼叫同一個語言核心生成`final_trace`。這是本小世界採用的回填格式；B章介紹的tool角色可以在未來擴充成更完整的對話協定。只有最後回答正常結束且內容正確，才算整題成功。若工具名正確、數字錯誤，工具會算另一題；若數字正確、回覆錯誤，仍是失敗。這些例子都要留在報告裡。

本輪動作評分還要求數字順序符合請求規格：「請算2加1」要生成`2+1`。`1+2`在數學上同樣得到3，但沒有照這份參數順序規格。報告的`action_correct`比對第一段完整字串與EOS，包含TOOL工具名與參數、或DIRECT／ASK的內容，不是只選對動作卡；`answer`與`expected_final`則讓你另查最後內容。端到端分數要求首段吻合且最後回答正確，不能把它直接叫作數學能力。

SFT驗證中的「4+4等於多少？」真的走完了這條迴圈：模型先產生`TOOL:calculator:4+4`，程式實際回傳`8`，同一模型讀回結果後產生`DIRECT:8`，兩次生成都正常結束。它與19.4那個同題失敗形成對照；這裡的最後8是模型生成的內容，沒有用程式結果冒充模型回答。

| SFT驗證的工具關卡 | 通過／該關卡總數 |
| --- | --- |
| 動作、工具名、數字及順序完整吻合 | 10／10個計算請求 |
| 真正執行計算器並收到成功結果 | 10／10個計算請求 |
| 同一模型讀回結果，正常結束且答對 | 10／10個計算請求 |
| 工具關閉時產生ASK且沒有執行 | 10／10個不可用情境 |
| 單獨提供工具結果後回答 | 5／5個回填題 |

最後一列是獨立的回填題，不能加到10個完整迴圈裡說成15個。數字家族雖與訓練分開，問法仍是固定模板；這些結果不代表任何自然語言計算題都能正確選工具。[SFT完整紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/sft/validation.json)保留請求、執行結果與兩次生成，可逐項核對。

joint共同訓練版與後續DPO分支在同一份驗證中也保留了這些工具行為。DPO是用[好壞回答配對](7.md#7.12)做偏好訓練的方法。這裡沒有觀察到新增圖音輸入破壞固定題型的工具流程，但圖音能力仍需分開檢查。定版後的各項工具分母統一見[19.12](19.md#19.12)。

最後檢查真的出現了驗證沒看到的讀回失敗。推薦joint對12個計算請求都產生正確工具名、數字與順序，程式也全部執行成功，最後同一模型卻只答對10／12。兩次失敗如下，兩次最後生成都有EOS，所以不是只差一個結束符號。

| 原問題 | 模型請求／工具真結果 | 模型讀回後生成 | 結果 |
| --- | --- | --- | --- |
| 0+1等於多少？ | `TOOL:calculator:0+1`／`1` | `DIRECT:0` | 工具正確，回答錯誤 |
| 請算1加0。 | `TOOL:calculator:1+0`／`1` | `DIRECT:111` | 工具正確，回答錯誤 |

另一方面，最後12個計算器關閉情境全部正確產生ASK且未執行；獨立回填題是5／6，其中「原題：0+1。計算器回報：1。請回答。」同樣答成0。預先留出的1加2與2加1則都請求正確、實際得到3、由模型最後答3。這些結果能區分「知道這類題該用工具」與「可靠讀回任何結果」：前者在此通過12／12，後者仍有失敗。

[正式逐題工具紀錄](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/test-joint.json)保留請求、實際結果與最後生成；[原始資料](https://github.com/birdhackor/tiny-perceptron-vlm/blob/1df335318bda03fd771807f66976953231d5a00b/docs/course-experiments/capstone-evidence/deployment/data.json)可按`id`查問句。這仍是固定加法模板的工具策略，不是模型已會校準所有任務的不確定性。

練習只把請求中的calculator改成unknown。先預測解析仍能讀懂一張工具請求，但執行器會拒絕未允許的工具，再核對。能理解格式與獲准執行是兩層檢查，不應該讓生成文字自己決定許可權。

