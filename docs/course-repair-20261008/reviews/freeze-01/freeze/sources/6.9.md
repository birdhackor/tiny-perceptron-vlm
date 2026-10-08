## 6.9 讀檔分段會改變切詞嗎？

hello一次編碼可合成一token；先讀hel再讀lo，各自編碼後接起卻可能多項。讀檔分塊只是一種儲存操作，不自動代表詞或文件邊界。

下面用純英文避開半個UTF-8字，先讓BPE從重複hello學合併，再比較整段與分段。

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(
    ["hello hello hello"],
    trainers.BpeTrainer(vocab_size=20, special_tokens=["[UNK]"], show_progress=False),
)
whole = tok.encode("hello").ids
chunks = tok.encode("hel").ids + tok.encode("lo").ids
print("整段", whole, "分段後相接", chunks)
assert whole != chunks
```

輸入訓練文字只反覆hello。`Whitespace` 先按空白分開詞，`BPE`再學常見組合；`[UNK]`是未知符號預留項，這個簡化英文表與 [6.3](#6.3) 的完整byte表不同。因為資料夠反覆，hello通常合成一token。whole應一個ID，chunks由至少兩段各自編碼再相接，因而長度更長、ID序列不同。精確編號只是約定，檢查的是整段與拆段的差別。

分段編碼不一定錯誤，關鍵是契約。如果每段本來就是不同文件，不能為縮短而跨文件合併；應明確保留開始、結束與文件邊界。若兩段屬於同一文件的連續材料，則應按整份文件編碼，或使用能保留尾段上下文、確認安全邊界才提交的流式方式。I/O chunk只是一塊讀入材料，不自動表示語義邊界。

同文件連續塊可先保留尾段、確認安全邊界才編碼；不同文件要保留邊界，不跨文合併。Whitespace與byte-level對空格的規則也不同。byte讀取若切半字，應保留尾bytes等下一塊，而不是先用�改原內容；兩層邊界都需守住。

練習用 `"hello hello"`，比較一次編碼與 `tok.encode("hello ").ids + tok.encode("hello").ids`。先預測這個Whitespace規則下兩者相同，再執行核對；然後恢復切在hel/lo中間，差異又出現。明示邊界與工具規則，才能解釋為何這次可以分段、上次不行。

<details>
<summary>補充：實作約定與原始紀錄</summary>

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>
