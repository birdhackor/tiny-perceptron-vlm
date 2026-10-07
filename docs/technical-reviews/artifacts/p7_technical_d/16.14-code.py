rules = {(): "今天", ("今天",): "晴天", ("今天", "晴天"): "〈E〉"}
draft = ["今天", "下雨", "嗎"]
accepted = []
for candidate in draft:
    target_next = rules[tuple(accepted)]
    if candidate != target_next:
        accepted.append(target_next)
        break
    accepted.append(candidate)
    if candidate == "〈E〉":
        break
print("這輪接上的token", accepted)
