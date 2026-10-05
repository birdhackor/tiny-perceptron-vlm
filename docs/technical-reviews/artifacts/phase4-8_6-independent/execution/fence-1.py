requests = [
    {"task": "確認日期", "date": None},
    {"task": "確認日期", "date": "2026-10-03"},
]
for request in requests:
    if request["date"] is None:
        target = "你希望哪一天？"
    else:
        target = "已確認日期：" + request["date"]
    print(request, "→", target)
