from tiny_perceptron.natural_concepts import picture_order_report

report = picture_order_report()
print("不同的像素數", report["different_pixels"])
print("4乘4平均摘要相同", report["same_4_by_4_summary"])
print("完整像素序列相同", report["same_pixel_sequence"])
