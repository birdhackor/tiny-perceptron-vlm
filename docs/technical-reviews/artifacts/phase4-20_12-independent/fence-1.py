from tiny_perceptron.natural_concepts import text_error_report

reference = "請推薦不辣的晚餐。"
recognized = "請推薦辣的晚餐。"
report = text_error_report(reference, recognized)
print("最少編輯次數", report["edits"])
print("參考字數", report["reference_characters"])
print("CER百分比", round(report["cer"] * 100, 1))
print("兩份問題相同", reference == recognized)
