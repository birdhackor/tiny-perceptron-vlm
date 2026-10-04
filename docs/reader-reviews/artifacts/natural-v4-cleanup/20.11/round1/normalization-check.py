reference = "牛奶\n麵包"
predicted = "牛奶麵包"
print("移除全部空白後相同", "".join(reference.split()) == "".join(predicted.split()))
