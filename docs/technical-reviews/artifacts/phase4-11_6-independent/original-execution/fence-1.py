budget = 10000
plans = {
    "兩階段": {"描述對齊": 4000, "問答微調": 6000},
    "直接問答": {"描述對齊": 0, "問答微調": 10000},
}
for name, plan in plans.items():
    total = sum(plan.values())
    print(name, plan, "總有效位置", total, "符合預算", total == budget)
