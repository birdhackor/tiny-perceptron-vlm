from tiny_perceptron.natural_concepts import text_error_report

report = text_error_report("今天去臺北", "今天去台北")
print("完整相同", report["exact"])
print("最少編輯次數", report["edits"])
print("參考字數", report["reference_characters"])
print("CER", report["cer"])
