## 7.9 模型怎麼知道回答結束？

回答A之後應結束，普通句號卻可能還要接下一句。把專用EOS放進回答尾端，並作為有效下一項目標，模型才有機會學「在這裡結束」。

生成上限另限制最多寫多少新token。EOS是模型選的結束，上限是外部預算；以下先查回答尾目標，再給手寫候選[A,EOS,B]，只驗證停止判斷。

```python
from tiny_perceptron.data import ByteTokenizer, render_chat

tok = ByteTokenizer()
x, y = render_chat([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
print("最後有效目標", y[-1].item(), "EOS", tok.eos_id)
assert y[-1].item() == tok.eos_id
generated = [tok.encode("A")[0], tok.eos_id, tok.encode("B")[0]]
max_new_tokens = 2
visible = []
for token in generated[:max_new_tokens]:
    if token == tok.eos_id:
        break
    visible.append(token)
print("顯示的內容", tok.decode(visible))
```

輸入短對話，render的最後標籤應2，是專用EOS。後半段用預先給定的模擬預測 `[73,2,74]`，不是模型訓練後實測；73表示A，74表示B。迴圈最多檢查兩項，每次若見EOS就 `break` 退出循環，不把它當正文加入visible。所以輸出內容只有A，後面的B不會顯示。

實際模型每次會替各候選下一ID給出分數，這組分數叫logits；接著選下一ID，再檢查是否停止，如同 [1.14](01.md#1.14) 的逐項生成。示例先給定結果，是為了單獨驗證停止規則。EOS通常留在保存的結構序列中，文字解碼時跳過這些專用標記；用戶看到的回答正文不需要顯示字面 `<eos>`。

歷史user段也可以有EOS，表示前段已結束；它不會讓當前assistant尚未生成就停止。逐byte生成上限還可能切半字，需要完整解碼規則。內容正確、模型選EOS、到上限截停應分開記，末尾有句號無法替它們判定。

練習把generated改為 `tok.encode("ABCD")`，保持max_new_tokens=2。先預測沒有EOS、只能顯示AB，再執行核對；這次停止來自上限，不代表模型學會適時結束。恢復原生成清單則顯示A，兩種停止來源都能直接驗證。

<details>
<summary>補充：實作約定與原始紀錄</summary>

實際學會結束，也不等於內容答對。[T.4的屬性問答模型](../training.md#T.4)訓練900次後，對一筆應回答`circle`的驗證題生成`cire`，原始ID是`[107,113,122,109,2]`。最後的2確實是EOS，前四項卻只拼出錯誤的答案。這份模型在全部五題驗證與十題最後檢查都主動結束，仍有許多內容錯誤；[完整實測報告](https://github.com/birdhackor/tiny-perceptron-vlm/blob/main/docs/course-experiments/results/sft.json)保留了逐題紀錄。因此評估要保留原始生成ID與EOS欄位，再分開計答案，不能只看最後有沒有停。

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

