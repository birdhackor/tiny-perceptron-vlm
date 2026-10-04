## B.6 模型如何學會選擇下一個動作？

上一節的動作卡是人填的。現在希望使用者只說「1加2等於多少」，模型就能根據問題與工具狀態選卡。像學生先看老師如何分類，再練習替新題分類：資料要同時包含需要工具、可以直接回答和需要求助的題目。若每一筆示範都使用計算器，模型就沒有看到何時該停止要求工具。

前置是[B.5的三種動作](0B.md#B.5)、[7.3只教回答位置](07.md#7.3)與[7.11監督式微調](07.md#7.11)。監督式微調，也叫SFT，是用正確示範更新模型的可調數字。本節只訓練選卡；卡選成TOOL後，如何填工具名稱、參數、執行與回填，仍是[B.1–B.4](0B.md#B.1)的另一段工作。

先看一筆可供訓練的對話。system放共同規則與工具狀態，user放自然提問，assistant放我們希望學到的動作。下面先檢查答案真的被放在會計算猜錯代價的位置，還沒有執行訓練。

```python
from tiny_perceptron.data import ByteTokenizer, render_chat

tokenizer = ByteTokenizer()
messages = [
    {"role": "system", "content": "精確計算用工具，解釋或照抄直接回答；缺資訊或工具先求助。計算器可用。"},
    {"role": "user", "content": "1加2等於多少？"},
    {"role": "assistant", "content": "TOOL"},
]
x, labels = render_chat(messages, tokenizer)
answer_ids = labels[labels != -100].tolist()
print("輸入位置數", len(x))
print("參與答案代價的文字", tokenizer.decode(answer_ids))
print("參與答案代價的位置數", len(answer_ids))
print("最後的標籤是EOS", answer_ids[-1] == tokenizer.eos_id)
```

ByteTokenizer將文字按UTF-8位元組編成ID，ID是字表索引，詳見[6.1](06.md#6.1)。本例沒有把常見片段合併，所以TOOL的四個英文字母各佔一個位元組位置。`render_chat`建立輸入x與答案標籤labels；-100表示該位置不直接計算答案代價，規則見[7.3](07.md#7.3)。`labels != -100`挑出有效標籤，`.tolist()`把它們轉成Python清單，再用decode還原文字。應看到文字TOOL、五個有效位置與最後一行True：四個英文字母和一個EOS。EOS是回答結束的特殊標記，decode還原文字時會略過它；清單的`[-1]`取最後一項，所以最後一行用它核對EOS。這五處提供直接的答案學習訊號，問題仍會經由前文影響預測。

真正訓練時，同樣的介面會反覆讀不同題目，更新模型，使合適動作更容易被產生。工具可用、停用的例子要成對呈現；概念解釋、照抄數字、缺資料的例子也要保留。使用者問題裡不加CALC或COPY提示，否則可能只教會讀前綴。模型仍可能只記住問法，所以[B.7](0B.md#B.7)需要另外留出新數字與新措辭。

我們另外從隨機權重訓練了一個選卡模型，並保存資料、設定、權重與原始生成。它沒有接續先前的工具模型，也沒有把新選卡器與舊JSON生成模型串接成已驗證的助手。完整流程可在已依[W.1](../first-steps.md#W.1)準備的專案根目錄執行；網站這段短程式不會偷偷開始長訓練：

```bash
python scripts/course_experiments/run.py --experiment tool_choice --device cpu
```

本輪固定配方與結果將在[B.7](0B.md#B.7)一起閱讀。這類示範教的是可觀察的工作策略，不是證明模型具有「我不擅長數學」的自我認知。研究中的[Toolformer](https://arxiv.org/abs/2302.04761)也研究語言模型如何學習工具呼叫；它使用不同的資料建立、篩選與訓練流程，本節沒有重做該論文的方法或成績。

練習把assistant內容由TOOL改成ASK，其他欄位不動。先預測有效位置變為四個：三個字母與EOS，再執行核對。這只改了示範答案，不會自行判斷新標註是否合理；根據system說計算器可用且題目數字齊全，這個ASK反而是不符合策略的示範。

