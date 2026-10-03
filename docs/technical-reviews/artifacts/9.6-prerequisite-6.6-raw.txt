## 6.6 邊界標記怎麼保持完整？

一份對話要區分「是誰說話」，不能只靠普通句子裡的文字。用戶也許真的想討論 `<assistant>` 這串字，如果程式看到拼寫就當成換角色，會把內容誤作結構。我們要讓角色、開始、結束與模態邊界有獨立ID，普通內容即使拼寫一樣仍按文字編碼。這裡角色指資料中的user、assistant等欄位，不是模型猜出講話人。

本專案簡單ByteTokenizer保留ID0到7做結構：PAD填充、BOS開始、EOS結束，接著user、assistant、image、audio、system。PAD是為了把不同長度排成同一張表的空位；image、audio表示後面有影像或音訊輸入。原文byte則統一加8，所以0至255的byte用8至263的ID，與結構區完全不重疊。

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

另一類切詞方法叫BPE：把經常相鄰的片段合成一個新單位，例如把`a`與`b`合成`ab`，詳見[6.2](#6.2)。這類工具通常允許先登記特殊標記，約定整串`<assistant>`使用一個專用ID，不被拆開，也不與旁邊的普通文字再合併。登記後仍要決定用戶內容中的相同拼寫怎樣處理，不能以為登記本身就自動解決「內容還是結構」的判斷。當前byte版本的偏移區分很直白，適合作為對話格式的起點。

這些標記不代替模型訓練。插入assistant ID只是把位置標出來，模型還要看很多正確示例才知道之後怎樣回答；image標記也不會自己創造影像特徵。它們是接口約定，讓資料與模型共享一種表示。

這次實跑的BPE也遵守相同區分：普通內容編碼不把特殊標記的字面拼寫升格成角色ID，結構仍由程式按資料欄位插入。實際核對`<user>這只是引用文字`時，它完整還原為同一段原文，內容ID中沒有任何八種控制ID。只把`<user>`登記成特殊項，卻讓內容編碼自動識別它，不能保證這個結果；使用工具時要連同「普通內容怎麼編碼」一起保存，不能只有一份詞表。

練習只把literal的輸入改成 `<image>`，並把檢查改為 `tok.image_id not in literal`。先預測還原仍是這七個字符、內容ID都至少8，不出現專用image ID5，再執行核對。只有工具顯式插入5才表示影像邊界，字面討論標記仍是一段文字。

