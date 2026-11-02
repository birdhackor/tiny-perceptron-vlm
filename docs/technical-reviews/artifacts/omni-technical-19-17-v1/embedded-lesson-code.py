# 節錄自 tiny_perceptron/lesson_products/inference.py：rule_depth
def rule_depth(policy, task, event_count=0):
    if policy.get("version") != 1:
        raise ValueError("需要凍結的Auto規則版本")
    if task == "sum":
        depth = policy["sum_depth"]
    else:
        depth = 1 if event_count <= policy["low_max_events"] else 3 if event_count <= policy["medium_max_events"] else 5
    if depth not in (1, 3, 5):
        raise ValueError("Auto規則只能選1／3／5輪")
    return depth

from tiny_perceptron.lesson_products.inference import EFFORT

policy = {"version": 1, "low_max_events": 0, "medium_max_events": 4, "sum_depth": 1}
for events in range(1, 7):
    print("兩盒追蹤", events, "個事件 → auto 選", rule_depth(policy, "boxes", events), "輪")
print("三數加法 → auto 選", rule_depth(policy, "sum"), "輪")
print("手動檔位", EFFORT)
print("指定 medium、硬上限 2 → 實跑", min(EFFORT["medium"], 2), "輪")
