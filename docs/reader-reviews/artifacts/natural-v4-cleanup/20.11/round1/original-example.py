from collections import Counter

reference = "牛奶\n麵包"
predicted = "麵包\n牛奶"
print("字元種類與數量相同", Counter(reference) == Counter(predicted))
print("完整字串相同", reference == predicted)
print("參考行序", reference.splitlines())
print("預測行序", predicted.splitlines())
