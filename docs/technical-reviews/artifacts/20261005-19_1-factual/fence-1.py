from tiny_perceptron.capstone import parse_action

examples = ["DIRECT:red", "ASK:請提供數量", "TOOL:calculator:1+2"]
for text in examples:
    action = parse_action({"raw": text, "eos": True})
    print(text, "→", action)
