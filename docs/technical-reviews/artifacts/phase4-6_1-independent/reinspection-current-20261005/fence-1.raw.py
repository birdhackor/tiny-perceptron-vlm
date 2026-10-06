texts = ["小小貓", "cat", "🙂"]
for text in texts:
    raw = text.encode("utf-8")
    print(text, "Unicode碼點", len(text), "UTF-8 bytes", len(raw))
