## 19.5 讓助理簡短、誠實與守規則，要教哪種示範？

先告訴助理「接下來請用兩點回答」，再問「App錯誤訊息一直重複出現。」本章公開 MoE 權重在 CPU 上真正生成了：

> 1. 先檢查網路並重新啟動App。  
> 2. 仍失敗再詢問官方客服。

這個回答的內容對應當前 App 問題，也符合先前要求的兩點格式。公開的[輸入訊息](../../docs/selftrained/examples/v2/text.messages.json)只含一般工作設定、格式確認與問題；目前的答案是在呼叫模型後才產生的。這是挑選出的成功驗證例，可以核對介面與生成流程，不能由一題推出所有新問法都成功。

要教出這種行為，示範不能只有「問題 → 一句好答案」，還要讓**條件改變時，答案也改變**。本輪文字訓練資料包含下列實際示範；表中是作者提供的訓練答案，與上方模型生成的回答用途不同。

| 工作條件與使用者問題 | 訓練示範回答 | 要教的差別 |
| --- | --- | --- |
| 用一句回答；「App打不開，該怎麼辦？」 | 先檢查網路並重新啟動App，仍失敗再詢問官方客服。 | 回應已教過的需求，使用指定格式 |
| 「那個問題要怎麼處理？」 | 請說明是地址、App還是卡片的問題。 | 資訊不足時先問，不自行補一個問題 |
| 計算器不可用；「算0加0是多少。」 | 計算器目前不可用，請開啟後再計算。 | 依工具狀態回答，不假裝已執行 |
| 「請預測明天天氣。」 | 這超出我的範圍，我能協助有限聊天、地址、App、卡片或加減乘問題。 | 說明這個有限助手的工作範圍 |

這裡的「誠實」有具體教法：缺條件就詢問、工具未開就說未開、範圍外就說明範圍。也要保留能直接回答的題目，否則每題都求助也可能看似謹慎。學會幾種澄清句，不等於能估計任意問題的可靠程度；成品在留出題上是否選對行為，仍要另外評分。

**監督式微調（SFT）**就是用這種示範調整參數，讓模型在相同前文下較可能生成示範回答。system 訊息寫一般工作條件，user 訊息寫請求，assistant 訊息寫希望學到的回應。本輪 SFT 更新自己的文字核心；問題與歷史仍在輸入裡，但作答代價只計 assistant 的文字與結束標記 EOS。EOS 表示這輪回答結束，也需要學，不能只學會前幾個字。

下面在 CPU 上整理一筆手寫示範，查看哪些字被列為 SFT 目標。它使用成品相同的訊息編碼器，沒有下載權重、生成回答或更新模型。

```python
from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer

messages = [
    {"role": "user", "content": "那個問題要怎麼處理？"},
    {"role": "assistant", "content": "請說明是地址、App還是卡片的問題。"},
]
tok = CharacterTokenizer.build([m["content"] for m in messages])
row = {"id": "teaching-demo", "task": "text", "messages": messages}
encoded = RecordEncoder(tok, ".").encode(row)
targets = [value for value in encoded["labels"] if value != -100]
print(tok.decode(targets, skip_special_tokens=False))
```

輸出是 `請說明是地址、App還是卡片的問題。<eos>`。`-100` 表示該位置不計作答代價，並沒有把使用者問題從輸入刪掉。真正訓練時，模型讀到問題與已給定的回答前綴，逐字預測下一字；真正推論時則要接著自己剛生成的字繼續回答。能把標籤整理對，只證明訓練目標的安排，不證明模型已學會。

文字示範還包含選項記憶、格式改寫與工具請求／讀回。訓練、驗證與最後題依前面的資料家族分開；驗收時分別查內容、指定範圍、格式、完整結束及澄清行為。[19.12](19.md#19.12)會列出分項結果。下一節沿用這些回答規則，只把其中一輪的文字材料換成圖片或聲音。

<details>
<summary>查證：示範、實際生成與成熟方法</summary>

訓練示範保存在固定資料包的 `text-tools-train.jsonl`；讀取與標籤安排見[成品資料編碼器](../../tiny_perceptron/selftrained/dataset.py)，各階段更新範圍見[訓練程式](../../scripts/selftrained/train.py)。上方 App 回答來自[公開 CPU 執行紀錄](../../docs/selftrained/infrastructure/v2-public-cpu-smoke-actual-review.json)，可重做的完整命令見[CPU 操作頁](../../docs/selftrained/v2-public-cpu-commands.md)。

[7.17](07.md#7.17)已解釋預訓練與示範微調的分工，第13章再介紹 PPO 與 DPO。這些是有用途與適用條件的成熟方法；本章成品的有限訓練不需要重新實驗每一種方法，才准許教材介紹它們。

</details>

