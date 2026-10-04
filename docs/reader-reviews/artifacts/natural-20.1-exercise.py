history = [{"role": "user", "content": "接下來請用繁體中文回答。"}]
typed_question = "請讀出照片裡的招牌。"
recognized_question = "請介紹這張照片。"
typed_chat = history + [{"role": "user", "content": typed_question}]
spoken_chat = history + [{"role": "user", "content": recognized_question}]
print("送進聊天模型的對話相同", typed_chat == spoken_chat)
print("最後一個問題", spoken_chat[-1]["content"])
print("共同的照片", "同一份圖片像素，不把文字標籤冒充圖片")
