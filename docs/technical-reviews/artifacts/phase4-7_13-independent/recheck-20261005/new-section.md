## 7.13 後續改動要和什麼比？

1+1的回答是{"answer":3}，格式整齊、數字卻錯。後續改风格、拒答或壓縮時，需先保存固定題和分項規則，才能知道改善的是交付方式，還是回答內容。

JSON把欄位名稱與值保存為文字。下面要求只有answer且值真的是整數，內容則另問是否等於2；這是兩個可以同時一過一錯的檢查。

```python
import json

reply = '{"answer": 3}'
parsed = json.loads(reply)
scores = {
    "格式符合": isinstance(parsed, dict) and set(parsed) == {"answer"} and type(parsed["answer"]) is int,
    "1+1內容正確": parsed.get("answer") == 2,
}
print(scores)
assert scores["格式符合"] and not scores["1+1內容正確"]
```

輸入reply是示意回答字串，不是現場模型生成。解析後得到字典，`isinstance(parsed, dict)` 檢查它是不是字典；這裡只接受恰好一個answer字段且值為整數。`type(value)` 取得值的類型，`type(value) is int` 要求它的類型確實是Python整數。JSON的 `true`、`false` 會解碼為表示真假的 `True`、`False`，這裡不把它們當作整數答案；若改用 `isinstance(value, int)`，它們卻也會通過，因此這裡需要更嚴格的檢查。`set(parsed)` 取得鍵集合，`get` 取answer欄位，若沒有則返回None；字典背景見 [W.2](../first-steps.md#W.2)。輸出格式True、內容False，最後檢查確認這兩種結果可以同時發生。

評估規則應在比較前寫明，例如算術用精確數值、分類用合法標籤；對開放式回答，可能需要多種合理答案或分項審讀，不一定只用字串完全相同。若只把所有表現加成一個總分，格式提升可能抵消事實退步；保存指標向量，也就是幾個分項數字，能保留這種行為差異。

基準保存完整題目、模型權重、匹配tokenizer和生成設定，後續改動用同一份題重測。正常結束也單獨記。反覆用基準選設定後它就是開發資料，最終仍需未參與選擇的新題。缺資訊、開放建議或安全邊界則各有相應標准，不能借JSON解析代判。

練習只把reply的3改成2，先預測格式仍True、內容變True，舊斷言應失敗，再執行核對並改成兩個都True的檢查。再把字段answer改名result，內容數字雖2，格式契約與字段讀取都不符；分別解釋這兩項失敗，讓評估規則可被追蹤。

<details>
<summary>補充：實作約定與原始紀錄</summary>

本課後續實驗保留的屬性基準是[T.4實跑](../training.md#T.4)的`sft/model.pt`，也就是[完整實測報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/sft.json)中從零練900次的模型，不是較好的`pretrain-sft.pt`。它在最後檢查只答對5／10，因此後續改動要對照這個實際起點，不能先假定舊能力已經滿分。原始報告每題保存問題、標準答案、生成ID、答案完全匹配與EOS；之後做新任務或壓縮時，沿用完整固定題目再量一次，才能看見哪些面向改善、哪些退步。

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

