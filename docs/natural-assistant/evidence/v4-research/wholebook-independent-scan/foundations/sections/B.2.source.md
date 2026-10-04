## B.2 文字怎麼變成有效呼叫？

模型傳來一段JSON文字，name寫add，a=2、b=3。我們不能只因它能被JSON讀懂就執行，還要確認這個名稱是系統提供的工具、欄位齊全、參數確實是數字。前置是[請求中的名稱與參數](0B.md#B.1)；Python字典與例外處理若不熟，可回看[最小Python操作](../first-steps.md#W.2)。本節把「讀出結構」與「允許執行」分成兩道檢查。

parse（解析）只把文字變成字典、清單或數字。`{"name":"delete_all","arguments":{"a":2,"b":3}}`是完整、可解析的JSON，但其中的delete_all不是本專案提供的工具。validate（驗證）再依約定核對內容：最外層只能有name、arguments；名稱只接受add與multiply；arguments只能有a、b，而且兩者都必須是數字。允許的名稱清單叫allowlist，欄位與型別規則則是這份請求的schema。

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

三個輸入都是可解析的JSON。第一行應印結果5；第二個因未知工具被拒絕；第三個因a是布林值被拒絕。`try`嘗試執行，`except ValueError`接住驗證錯誤並印原因，沒有把失敗偷偷當成成功。`call_tool`先解析字串，核對欄位和型別，通過後才算加法或乘法。這個例子完全在本地，不需要模型、網路或任意程式執行。

第三例特別容易漏掉：JSON的true會變成Python的True，而Python裡布林值是整數類型的子類，某些寬鬆檢查會把它當1。本工具用`type(v) in (int,float)`核對確切型別，所以不接受bool；把`"2"`寫成字串也會拒絕，不會偷偷當數字。這讓工具輸入有一致的解釋，模型不能靠含糊格式意外改變運算。

通過驗證只保證符合這份小工具的規則，不保證數字來自正確問題。若使用者問2+3而請求寫2+4，工具仍會合法回傳6；那是參數選擇錯誤，需對照原問題。正式記錄應包含原文字、解析後欄位、拒絕原因與執行結果，才能區分解析失敗、內容不允許和運算已完成三種情況。

先說明接下來會遇到的特殊數值。`1e308`是科學記數法，表示`1×10^308`；Python的float能保存它，但能保存的範圍有限。`1e999`表示`1×10^999`，超過float範圍後會變成Infinity（無限大），Python常印成`inf`。`NaN`是Not a Number的縮寫，表示「不是有效數字」的特殊值。「有限float」指既不是正負無限大，也不是NaN。

正式模型的外層協議在`call_tool`之前另做嚴格JSON解析：拒絕`NaN`、`Infinity`、轉成非有限float的`1e999`，也拒絕任何層級的重複欄位。這與上面的最小`call_tool`字串示範是不同入口，不能假定單用那段短程式就有全部檢查。只有解析成功且內容符合add／multiply白名單，才真正執行；非法控制token也在解析前拒絕。

報告另外保存七份人工故障注入，包括不完整JSON、清單、未知工具、bool、`NaN`、`1e999`與重複name，全部被拒絕。它們測的是驗證程式，沒有算進20題模型成績。有限輸入也可能乘到溢位，也就是結果超出float可保存的範圍；例如`1e308×1e308`的兩個輸入都有限，結果卻成為`inf`。協議遇到這種情況保留`executed=true`、`finite_result=false`、字串`"inf"`，以`invalid_tool_result`停止，真正呼叫次數仍算一次，不能把非有限數字寫進JSON。這條分支由[CPU回歸案例](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/tests/test_application_protocols.py)檢查，本輪小數字的模型測試沒有出現溢位。

練習只把第三份JSON的true改成2，保持b與name不變。先預測它從拒絕變為結果5，第二份仍被拒絕，再執行核對。接著說明為何這不是擴大了工具權限：允許的名稱和規則完全相同，只是輸入終於符合數字型別。

