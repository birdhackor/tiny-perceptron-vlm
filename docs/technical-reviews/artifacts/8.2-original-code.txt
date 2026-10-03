question = "2+2=?"
requests = ["用一句話回答", "用貼切比喻回答"]
for style in requests:
    prompt = style + "：" + question
    print(prompt)
print("兩次都使用同一份權重與同一個選字方式")
