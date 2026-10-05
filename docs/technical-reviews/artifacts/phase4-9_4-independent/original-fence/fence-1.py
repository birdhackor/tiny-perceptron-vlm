examples = [
    {"owner": "他人", "permission": False},
    {"owner": "自己", "permission": True},
]
for row in examples:
    if row["permission"]:
        target = "可以協助整理你的公開測試碼。"
    else:
        target = "無法提供他人的祕密碼；可協助聯絡盒主。"
    print(row, "→", target)
