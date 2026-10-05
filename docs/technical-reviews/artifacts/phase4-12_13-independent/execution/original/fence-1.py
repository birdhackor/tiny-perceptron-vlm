from tiny_perceptron.natural_concepts import audio_order_report

report = audio_order_report()
print("時間框與頻帶", report["time_frames"], report["bands"])
print("先後序列相同", report["same_time_sequence"])
print("平均摘要近似相同", report["same_mean_with_roundoff_tolerance"])
