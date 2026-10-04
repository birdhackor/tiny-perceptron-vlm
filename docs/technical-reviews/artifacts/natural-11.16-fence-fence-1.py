from tiny_perceptron.natural_concepts import text_error_report

report = text_error_report("今天去臺北", "今天去台北")
print("完全相同", report["exact"])
print("最少編輯次數", report["edits"])
print("參考字數與CER", report["reference_characters"], report["cer"])
