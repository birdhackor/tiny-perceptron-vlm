reference = "請算十三加二。"
recognized = "請算十三加二。"
original_numbers = (13, 2)
recognized_numbers = (13, 2)
print("逐字稿完整符合", reference == recognized)
print("原問題的兩數相加", sum(original_numbers))
print("錯誤逐字稿的兩數相加", sum(recognized_numbers))
print("這是人工漏字示例，沒有辨識錄音或呼叫聊天模型")
