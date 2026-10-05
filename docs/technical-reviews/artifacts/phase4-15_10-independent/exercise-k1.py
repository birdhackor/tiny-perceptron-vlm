width, hidden, k = 4, 8, 1
one_expert = width * hidden + hidden * width
for experts in (4, 8):
    router = width * experts
    total = experts * one_expert + router
    active = k * one_expert + router
    print("expert數", experts, "總權重", total, "每token使用估計", active, "FP32權重bytes", total * 4)
