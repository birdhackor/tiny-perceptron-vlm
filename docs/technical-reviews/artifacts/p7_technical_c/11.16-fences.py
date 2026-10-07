import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
from tiny_perceptron.natural_concepts import text_error_report

reference = "今天去臺北"  # 參考答案
prediction = "今天去台北"  # 預測結果
report = text_error_report(reference, prediction)  # 先參考、後預測
print("完全相同", report["exact"])
print("最少編輯次數", report["edits"])
print("參考字數與CER", report["reference_characters"], report["cer"])

