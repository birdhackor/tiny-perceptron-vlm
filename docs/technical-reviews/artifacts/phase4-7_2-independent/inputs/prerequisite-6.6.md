## 6.6 邊界標記怎麼保持完整？

使用者正在討論「`<assistant>`」這串字，它應保持內容，不該突然換角色。結構由role欄位插入，普通文字用不重疊ID區，兩種用途才能分開。

ByteTokenizer的ID0到7依序是PAD、BOS、EOS、user、assistant、image、audio、system；byte加8後用8到263。PAD排齊長度，開始、結束與模態項標結構，不是可讀內容。

```python
from tiny_perceptron.data import ByteTokenizer, SPECIALS

tok = ByteTokenizer()
print(list(enumerate(SPECIALS)))
literal = tok.encode("<assistant>")
print("普通文字的ID", literal)
print("還原", tok.decode(literal))
assert tok.assistant_id not in literal
```

輸入字串 `<assistant>` 是要討論的普通文字。第一行把八種結構拼寫與位置印出，assistant的專用ID為4。`encode` 僅把原文每個UTF-8 byte加8；所以ASCII的 `<` 原來60變68，末尾 `>` 原來62變70，所有內容ID都至少8，絕不含4。decode減回偏移並拼成bytes，再恢復原拼寫，最後檢查這串文字沒有被升格成assistant角色。

真正需要插入角色時，資料工具按結構化記錄的role欄位明確加入4，再附內容ID。例如record中的 `role="assistant"` 是元資料決定，不是掃描內容發現那個單詞就轉換。第 [7.1](07.md#7.1)、[7.2](07.md#7.2) 將把完整對話走一次。原文與結構分開，能保持邊界完整，也讓內容可無損還原。

真正助手邊界是工具按role插4再接內容。BPE可登记整串特殊項，仍须另外决定普通內容相同拼寫怎麼编碼。只保存特殊項不足以守住引用邊界；字表與普通／結構契約要一同保存。插標記也不會自行教會回答或創造影像特徵。

練習只把literal的輸入改成 `<image>`，並把檢查改為 `tok.image_id not in literal`。先預測還原仍是這七個字符、內容ID都至少8，不出現專用image ID5，再執行核對。只有工具顯式插入5才表示影像邊界，字面討論標記仍是一段文字。

<details>
<summary>補充：實作約定與原始紀錄</summary>

這次實跑的BPE也遵守相同區分：普通內容編碼不把特殊標記的字面拼寫升格成角色ID，結構仍由程式按資料欄位插入。實際核對`<user>這只是引用文字`時，它完整還原為同一段原文，內容ID中沒有任何八種控制ID。只把`<user>`登記成特殊項，卻讓內容編碼自動識別它，不能保證這個結果；使用工具時要連同「普通內容怎麼編碼」一起保存，不能只有一份詞表。

短程式沿用本課工具與原計算語義。安裝、長訓練與重做操作見[訓練配方](../training.md)，不需要先完成長配方才能閱讀這個例子。

</details>

