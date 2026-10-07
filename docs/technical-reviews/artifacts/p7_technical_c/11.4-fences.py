import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
records = [
    {"image": "red_circle", "question": "什麼顏色？", "answer": "紅"},
    {"image": "red_circle", "question": "什麼形狀？", "answer": "圓形"},
]
for row in records:
    print(row["image"], row["question"], "→", row["answer"])
print("同圖", records[0]["image"] == records[1]["image"])
print("答案不同", records[0]["answer"] != records[1]["answer"])

