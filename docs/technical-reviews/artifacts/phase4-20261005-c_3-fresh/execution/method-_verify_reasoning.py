def _verify_reasoning(sample, row, mode):
    text = sample["generated"].strip()
    clean = not sample["invalid_special_tokens"]
    if mode == "direct":
        parsed = int(text) if clean and re.fullmatch(r"-?[0-9]+", text) else None
        return {
            "final_answer": parsed,
            "parsed": parsed is not None,
            "final_correct": parsed == row["truth"],
            "fully_verified": parsed == row["truth"],
        }
    matched = re.fullmatch(
        r"(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)", text
    )
    if not matched or not clean:
        # 結尾格式可獨立解析，讓「最終對但過程錯」保留在資料中。
        final_match = re.search(r";answer=(-?[0-9]+)$", text) if clean else None
        parsed = int(final_match[1]) if final_match else None
        return {
            "final_answer": parsed,
            "parsed": False,
            "final_correct": parsed == row["truth"],
            "equations_valid": False,
            "linked": False,
            "task_operands_valid": False,
            "fully_verified": False,
        }
    a, b, subtotal, previous, c, total, final = map(int, matched.groups())
    equations = a + b == subtotal and previous + c == total
    linked = previous == subtotal and total == final
    task = (a, b, c) == (row["a"], row["b"], row["c"])
    return {
        "final_answer": final,
        "parsed": True,
        "steps": [[a, b, subtotal], [previous, c, total]],
        "final_correct": final == row["truth"],
        "equations_valid": equations,
        "linked": linked,
        "task_operands_valid": task,
        "fully_verified": equations and linked and task and final == row["truth"],
    }
