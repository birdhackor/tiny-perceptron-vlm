## B.6 模型如何學會選擇下一個動作？

目前卡是人填的；希望模型看到「1加2等於多少」與「計算器可用」，能生成TOOL。資料同時含計算、解釋、照抄、缺資訊，以及工具可用／停用對照，才教得到不同路。

先看一份對話如何讓動作成為學習目標：system放策略與狀態，user放問題，assistant放TOOL。

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

ByteTokenizer按UTF-8位元組（byte）編號，文字與位元組數的對照見[6.1](06.md#6.1)。TOOL四個字母加EOS，共五個有效目標。decode略過EOS，所以另查最後ID。-100只是問題等位置不計答案代價，問題仍可讀。這段只檢查資料，沒有更新模型。

真正選卡訓練更新可調參數，使合適標記更容易出現。自然問題不用CALC／COPY前綴，避免只學讀標頭。卡選TOOL後，名稱、參數、真執行與讀回另驗；單獨選卡模型沒有直接完成整個往返。

舊選卡器由隨機權重另建，與先前JSON模型不是同一權重。本節人工system寫完整規則幫助理解，實際舊訓練system只提供工具狀態，策略放在示範標籤。這種選擇是工作策略，不是模型具有「我知道我不會算」的自我認知。

<details>
<summary>補充：舊選卡器的重做與方法來源</summary>

我們另外從隨機權重訓練了一個選卡模型，並保存資料、設定、權重與原始生成。它沒有接續先前的工具模型，也沒有把新選卡器與舊JSON生成模型串接成已驗證的助手。完整流程可在已依[W.1](../first-steps.md#W.1)準備的專案根目錄執行；網站這段短程式不會偷偷開始長訓練：

```bash
python scripts/course_experiments/run.py --experiment tool_choice --device cpu
```

本輪固定配方與結果將在[B.7](0B.md#B.7)一起閱讀。這類示範教的是可觀察的工作策略，不是證明模型具有「我不擅長數學」的自我認知。研究中的[Toolformer](https://arxiv.org/abs/2302.04761)也研究語言模型如何學習工具呼叫；它使用不同的資料建立、篩選與訓練流程，本節沒有重做該論文的方法或成績。

</details>

