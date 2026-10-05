## 6.3 怎麼訓練實用的 tokenizer？

只用「貓看狗。」與「a cat sees a dog.」訂字表，還要表示新文字「小鳥🙂new」。起始表只收見過字可能漏新人名；先收完整256種byte，再合常見片段，未見文字就有細表示。

以下用tokenizers建立byte-level BPE，目標詞表280。訓練的是拆分規則；之後encode轉ID，decode拼回原文。

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=280,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
tok.train_from_iterator(["貓看狗。", "a cat sees a dog."], trainer)
text = "小鳥🙂new"
encoded = tok.encode(text)
print("片段", encoded.tokens)
print("ID", encoded.ids)
print("還原", tok.decode(encoded.ids))
assert tok.decode(encoded.ids) == text
```

`models.BPE` 選合併規則類型；`ByteLevel` 前處理把byte映射為工具內部可見符號，再給BPE統計。`initial_alphabet` 納入完整256種起始byte符號，而非只加入兩句出現的字符。目標詞表280留下少量合併空間；數據很少時未必能恰好長到目標上限。`add_prefix_space=False` 避免主動在開頭加空格，也沒有額外normalizer改原文。

decoder是反方向的解碼工具，按ByteLevel約定拼回bytes再轉UTF-8。輸入新句含訓練未見的小、鳥與🙂，它們仍能以細片段表示。印出的tokens可能像奇怪拉丁符號，因為展示的是byte的可見代號，不是損壞的原句；應核對最後「還原」行完整為 `小鳥🙂new` 與assert通過，而不要求每token肉眼可讀成完整字。

詞表需容起始byte、特殊項與合併項，資料少時未必剛好達280。實用大小還需量未見文字長度和模型成本。保存規則時記版本及ID含義；重訓相同大小字表，ID仍可能換意思，不能直接接舊模型。

練習只把text改成 `"未見字🦊"`，先預測可能切得較細，但decode仍等於原文，再執行核對。若你同時改了前處理規則，應重新驗證空格、換行與特殊字符；還原契約來自完整規則組合，不只BPE這個名字。

<details>
<summary>補充：實作約定與原始紀錄</summary>

這個短示範之後，我們也完成一次較接近資料準備流程的實驗：從固定資料包各取96篇英文故事與96首中文詩，先按完整篇章分側，再各取至多256個byte、末尾保持完整字的片段。BPE只用153篇訓練側片段學規則，實際字表為512項，包含八個邊界標記；保存成`tokenizer-bpe512.json`。沒有拿另外19篇驗證與20篇最後檢查來決定合併規則。未見文字`未見字🦊 new`、含頭尾空白和換行的` 小鳥🙂\nnew `都能完整還原；此處`\n`表示一個實際換行。這些核對支持表示工具的還原能力，不等於語言模型已懂句意；模型成績留到[6.5](#6.5)比較，重跑入口在[T.4](../training.md#T.4)。

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

