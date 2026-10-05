def _reasoning_sft(rows, mode):
    result = []
    for row in rows:
        a, b, c, truth = row["a"], row["b"], row["c"], row["truth"]
        answer = str(truth) if mode == "direct" else f"{a}+{b}={a + b};{a + b}+{c}={truth};answer={truth}"
        result.append({**row, "messages": _conversation(row["question"], answer)})
    return result
