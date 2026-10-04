from tiny_perceptron.retrieval import tool_loop

requests = [
    {"name": "add", "arguments": {"a": 2, "b": 3}},
    {"done": True, "answer": 5},
]
runs = [
    tool_loop(requests, max_steps=3),
    tool_loop(requests, max_steps=1),
    tool_loop(requests[:1], max_steps=3),
    tool_loop([{"name": "unknown", "arguments": {"a": 2, "b": 3}}]),
]
for run in runs:
    print(run["status"], "工具執行筆數", len(run["trace"]), "答案", run.get("answer"))
