## B.6 模型如何學會選擇下一個動作？

上一節的卡是人填的。現在希望模型看到「1 加 2 等於多少？」與「計算器可用」時，自己生成 TOOL。資料要同時包含計算、解釋、照抄、缺資訊，以及同一題在計算器可用與停用時的對照，才教得到不同的選擇。

下面是一筆訓練資料的樣子，格式與選卡實驗的資料相同：system 只寫工具狀態，user 放問題，assistant 放正確的卡。決策規則本身不寫進 system；模型要從許多示範中學到「什麼問題、什麼狀態配哪張卡」。

```python
from tiny_perceptron.data import ByteTokenizer, render_chat

tokenizer = ByteTokenizer()
messages = [
    {"role": "system", "content": "計算器可用。"},
    {"role": "user", "content": "1 加 2 等於多少？"},
    {"role": "assistant", "content": "TOOL"},
]
inputs, labels = render_chat(messages, tokenizer)
answer_ids = labels[labels != -100].tolist()
print("輸入位置數", len(inputs))
print("有效目標解碼成文字", tokenizer.decode(answer_ids))
print("有效目標數", len(answer_ids))
print("最後一個目標是EOS", answer_ids[-1] == tokenizer.eos_id)
```

`render_chat`（實作見 [7.1](07.md#7.1)）把三則訊息排成模型輸入 `inputs` 與目標 `labels`；`ByteTokenizer`（[6.6](06.md#6.6)）按 UTF-8 位元組（byte）編號，文字與 byte 數的對照見 [6.1](06.md#6.1)。`labels[labels != -100]` 只留下有效目標：−100 標示不計代價（loss）的位置（7.1）。輸入共 51 個位置；有效目標是 T、O、O、L 四個字母加上 EOS，共 5 個。`tokenizer.decode` 解碼時會略過 EOS 這類特殊 token，所以解碼出的文字只有 TOOL，最後一行另外確認最後一個目標確實是 EOS。system 與 user 的內容不計答案代價，但仍留在輸入裡讓模型讀。這段只檢查資料，沒有更新模型。

真正的選卡訓練會更新模型的參數，讓合適的卡更容易被生成。自然問句不加 CALC、COPY 這類前綴，避免模型只靠前綴猜答案。選到 TOOL 之後，請求裡的 a、b 是否正確、工具是否真正執行、最後回答是否正確，都還要另外檢查；只學選卡的模型，還不能自己完成整個流程。

選卡實驗的模型從隨機參數另外訓練，和 B.1 工具實驗的 JSON 模型不是同一份參數。它學到的是人訂的決策規則，不是「我知道自己不會算」的自我認知。

練習把 system 改成 `"計算器停用。"`、assistant 改成 `"ASK"`，先預測：「停用」與「可用」一樣是兩個字，ASK 卻比 TOOL 少一個字母，所以輸入位置數變成 50、有效目標數變成 4，解碼出 ASK；再執行核對。

<details>
<summary>補充：選卡實驗的重做與方法來源</summary>

選卡實驗另外從隨機參數訓練了一個選卡模型，並保存資料、設定、參數與原始生成。它沒有接續 B.1 的工具實驗模型，也沒有和那個 JSON 生成模型串成已驗證的助理。完整流程可以在已依 [W.1](../first-steps.md#W.1) 準備好的專案根目錄執行；網站上的短程式不會偷偷開始長時間的訓練：

```bash
python scripts/course_experiments/run.py --experiment tool_choice --device cpu
```

固定的配方與結果在 [B.7](0B.md#B.7) 一起讀。這類示範教的是看得見的決策規則，不能證明模型有「我不擅長數學」的自我認知。研究中的 [Toolformer](https://arxiv.org/abs/2302.04761) 也研究語言模型如何學會呼叫工具；它的資料建立、篩選與訓練流程都不同，本節沒有重做該論文的方法或成績。

</details>

