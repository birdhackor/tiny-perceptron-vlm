## 8.5 答案正確，為什麼JSON仍可能不合格？

要求「2+2=? 只回JSON，只有answer欄位」，合格文字是`{"answer":4}`。`答案是：{"answer":4}`雖然含正確數字，接收程式卻不能把整份回答直接當JSON讀取；`{"answer":5}`能讀取，數字卻錯。本節把這兩道檢查分開。

Python的`json.loads`將JSON字串讀成資料。**解析**就是這個讀取步驟；格式破損或多出散文時，它會報錯。讀取成功後，還要核對欄位與值，才能判斷是否完成要求。

```python
import json

answers = ['{"answer":4}', '答案是：{"answer":4}', '{"answer":5}']
for text in answers:
    try:
        value = json.loads(text)
        valid = isinstance(value, dict) and set(value) == {"answer"}
        correct = valid and value["answer"] == 4
    except json.JSONDecodeError:
        valid = correct = False
    print(text, "格式", valid, "內容", correct)
```

三份手寫回答輸出依序為`格式 True 內容 True`、`False False`、`True False`。`try`先嘗試讀取，`except`接住解析錯誤，讓下一筆仍能繼續檢查。`isinstance(value, dict)`確認讀到字典；`set(value)`收集欄位名稱，與只含answer的集合比對，排除漏欄位或多欄位。

本例「內容」欄的定義是：格式通過後，answer的值還必須等於4。因此第二份雖可看見4，仍記為不能交付的內容。分項的名稱與判準要一起保存，避免誤以為這段程式在理解所有自然語句。若任務還規定欄位值的型別，應另加相應檢查。

練習把第一份改成`{"answer":4,"note":"完成"}`。解析會成功，但多出的note違反「只有answer欄位」，兩項都記`False`。如果希望接受note，應先改任務和檢查規則。這裡三筆都是手寫對照；模型實際生成後，才把原回答送入同一檢查，不能先替它刪掉多出的話再算成功。

<details>
<summary>補充：既有實驗的條件與完整紀錄</summary>

上一節的模型把這個反例真的生成了：七道最後檢查的JSON都可解析、欄位與整數類型也正確，但其中的加法數值都錯。程式能讀取回答，只是第一道門；有明確真值的任務還要按題目驗證值。這也解釋為什麼訓練與評估要保留分項，不能只說「JSON成功率很高」。

</details>

