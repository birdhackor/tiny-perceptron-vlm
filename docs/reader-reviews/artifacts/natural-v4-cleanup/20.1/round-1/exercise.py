history = [{"role": "user", "content": "接下來請用繁體中文回答。"}]
typed_question = "請讀出照片裡的招牌。"
recognized_question = "請介紹這張照片"
typed_chat = history + [{"role": "user", "content": typed_question}]
spoken_chat = history + [{"role": "user", "content": recognized_question}]
print("聊天模型收到相同對話", typed_chat == spoken_chat)
print("這輪問題", spoken_chat[-1]["content"])
