## B.2 文字怎麼變成有效呼叫？

一段JSON寫add、a=2、b=3，先讀成欄位，再依工具規則核對，通過才執行。解析是讀結構，驗證是查允許內容；格式完整的delete_all仍不在本例工具清單裏。

本例只允許add與multiply，外層欄位只有name、arguments，參數只有數字a、b。名稱清單叫allowlist，欄位與型別契約叫schema。

```python
from tiny_perceptron.retrieval import call_tool

requests = [
    '{"name":"add","arguments":{"a":2,"b":3}}',
    '{"name":"delete_all","arguments":{"a":2,"b":3}}',
    '{"name":"add","arguments":{"a":true,"b":3}}',
]
for text in requests:
    try:
        print("結果", call_tool(text))
    except ValueError as error:
        print("拒絕", error)
```

第一份真的算出5，第二份未知工具被拒，第三份bool被拒。Python的True雖屬int子類，本工具用確切型別 `type(v) in (int,float)`，所以不把true當1，也不把字串"2"當數字。

通過只說明契約合法，仍要回查原題。原題2+3、請求2+4也會合法回6；它是模型選錯參數，不是程式算錯。解析、驗證、實際執行各留紀錄，才能分清錯在哪一站。

正式協議在最小 `call_tool` 前還做嚴格JSON檢查，拒絕NaN、Infinity、轉成非有限值的1e999及重複欄位；不能把這項保證直接套給單用短例的入口。有限輸入也可能使乘法溢位，結果不是有限數字時要記已執行與失敗原因，不假冒成功。

<details>
<summary>補充：數值範圍與故障注入</summary>

先說明接下來會遇到的特殊數值。`1e308`是科學記數法，表示`1×10^308`；Python的float能保存它，但能保存的範圍有限。`1e999`表示`1×10^999`，超過float範圍後會變成Infinity（無限大），Python常印成`inf`。`NaN`是Not a Number的縮寫，表示「不是有效數字」的特殊值。「有限float」指既不是正負無限大，也不是NaN。

正式模型的外層協議在`call_tool`之前另做嚴格JSON解析：拒絕`NaN`、`Infinity`、轉成非有限float的`1e999`，也拒絕任何層級的重複欄位。這與上面的最小`call_tool`字串示範是不同入口，不能假定單用那段短程式就有全部檢查。只有解析成功且內容符合add／multiply白名單，才真正執行。控制 token 是角色或序列邊界的專用標記（見[7.2](07.md#7.2)）：例如工具請求中混入專用 user 角色標記，它不是JSON字詞，會在解析前拒絕。正常生成結束的 EOS 則是停止標記（見[7.9](07.md#7.9)）。

報告另外保存七份人工故障注入，包括不完整JSON、頂層陣列（如 `[]`）、未知工具、bool、`NaN`、`1e999`與重複name，全部被拒絕。這裡的工具請求須是帶 name、arguments 的JSON物件，不能把一個陣列當單份請求；它與Python用來依次試三份文字的 `requests` 清單不同。它們測的是驗證程式，沒有算進20題模型成績。有限輸入也可能乘到溢位，也就是結果超出float可保存的範圍；例如`1e308×1e308`的兩個輸入都有限，結果卻成為`inf`。協議遇到這種情況保留`executed=true`、`finite_result=false`、字串`"inf"`，以`invalid_tool_result`停止，真正呼叫次數仍算一次，不能把非有限數字寫進JSON。這條分支由[CPU回歸案例](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/tests/test_application_protocols.py)檢查，本輪小數字的模型測試沒有出現溢位。

</details>

