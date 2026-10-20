T = 16
prompt = ["〈user〉", "店名是小小書店。", "〈eos〉", "〈assistant〉", "好。", "〈eos〉"]
prompt += ["〈user〉", "現在", "只答", "店名。", "〈eos〉", "〈assistant〉"]
P = len(prompt)
print("前文長度P", P, "還能生成的位置T-P", T - P)
candidates = ["小小", "書店", "〈eos〉", "謝謝"]
for G in [4, 2, 5]:
    if P + G > T:
        print(f"G={G}：P+G={P + G} 超過 T={T}，拒絕這個設定")
        continue
    reply = []
    for token in candidates[:G]:
        reply.append(token)
        if token == "〈eos〉":
            break
    reason = "模型選了EOS" if reply[-1] == "〈eos〉" else "用完G"
    print(f"G={G}：新增{len(reply)}個token {reply}，停止原因：{reason}")
