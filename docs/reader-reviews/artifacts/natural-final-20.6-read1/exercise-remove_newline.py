from collections import Counter

reference = "牛奶\n麵包"
predicted = "牛奶麵包"
print("同樣字元與數量", Counter(reference) == Counter(predicted))
print("完整字串相同", reference == predicted)
print("參考行序", reference.splitlines())
print("預測行序", predicted.splitlines())
