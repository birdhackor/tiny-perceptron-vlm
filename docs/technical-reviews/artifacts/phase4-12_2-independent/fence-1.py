samples = 1600
for rate in [16000, 8000]:
    print("取樣率", rate, "時長", samples / rate, "奈奎斯特界線", rate / 2, "440Hz每週期樣本", round(rate / 440, 2))
