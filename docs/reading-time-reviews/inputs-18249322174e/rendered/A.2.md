# A.2：網站實際顯示的範例程式與 CPU 輸出

## 範例 cell 4

```python
source = {"id": "公告1", "date": "2027年3月1日", "address": "青街8號"}
context = f"小小書店在{source['date']}搬到{source['address']}。"
prompt = f"[來源{source['id']}] {context}\n問題：書店新地址？請依資料回答並標示來源。"
print(prompt)
print("由資料確定的答案", source["address"])
```

```text
[來源公告1] 小小書店在2027年3月1日搬到青街8號。
問題：書店新地址？請依資料回答並標示來源。
由資料確定的答案 青街8號

```

