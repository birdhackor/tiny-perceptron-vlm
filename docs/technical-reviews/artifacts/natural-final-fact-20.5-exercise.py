truth = {"objects": {"人", "腳踏車"}, "activity": "騎車"}
cards = [
    {"objects": {"人", "腳踏車"}, "activity": "騎車"},
    {"objects": {"人", "腳踏車"}, "activity": "站在車旁"},
]
cards[1]["objects"].add("狗")
for index, card in enumerate(cards):
    print("卡", index, "物件集合吻合", card["objects"] == truth["objects"])
    print("活動符合本題標註", card["activity"] == truth["activity"])
