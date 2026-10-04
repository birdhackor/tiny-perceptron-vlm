from tiny_perceptron.capstone import calculator_runtime, parse_action

trace = {"raw": "TOOL:calculator:1+2", "eos": True}
action = parse_action(trace)
print("解析後的請求", action)
for available in (True, False):
    result = calculator_runtime(action, available=available)
    print("計算器可用", available, "實際執行結果", result)
print("還沒有模型讀回結果，也沒有最終模型答案")
