## 20.1 照片、打字與說話，怎麼交給同一位助理？

先打字說「接下來請用繁體中文回答」，附圖問招牌，下一輪改用聲音追問。我們希望照片與語言要求仍保留，換的是這輪問題的入口。成熟模型延伸採同一份聊天歷史：打字直接加入；錄音先由 Whisper 聽寫，再把逐字稿加入同一位置。

ASR 是語音辨識，輸出「聽到哪些字」，不是問題的答案。圖文核心 Qwen 讀照片、歷史與這輪文字，才生成回答。可選 LoRA 是核心旁的小修正，並不是另一位聊天模型；目前交付配置沒有開啟它。

![打字或Whisper逐字稿接到同一份歷史，與照片一起由Qwen回答。](../figures/rewrite-20-input-routes.svg)

```python
history = [{"role": "user", "content": "接下來請用繁體中文回答。"}]
typed_question = "請讀出照片裡的招牌。"
recognized_question = "請讀出照片裡的招牌。"
typed_chat = history + [{"role": "user", "content": typed_question}]
spoken_chat = history + [{"role": "user", "content": recognized_question}]
print("聊天模型收到相同對話", typed_chat == spoken_chat)
print("這輪問題", spoken_chat[-1]["content"])
```

這兩份問題由人手寫成相同，所以對話比較為 True。這段只示範會合點，未辨識錄音或讀圖。真正兩路比較還要固定照片、歷史與生成設定，查看ASR是否把條件改掉。

介面先顯示辨識原稿，使用者可以更正後再送出。評測原始語音路線時保留更正前的逐字稿；更正後成功是人工操作結果，不能偷偷算成辨識器原本就聽對。這套配置以文字回答，沒有因此具備語音朗讀。

