groups = {"常見": {"correct": 90, "count": 90}, "稀少": {"correct": 0, "count": 10}}
micro = sum(g["correct"] for g in groups.values()) / sum(g["count"] for g in groups.values())
macro = sum(g["correct"] / g["count"] for g in groups.values()) / len(groups)
for name, group in groups.items():
    print(name, "答對/總數", group["correct"], group["count"])
print("按題平均", micro, "按組平均", macro)
