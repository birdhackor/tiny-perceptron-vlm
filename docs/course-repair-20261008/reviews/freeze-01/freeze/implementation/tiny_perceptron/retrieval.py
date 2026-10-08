"""跳讀單元的本地檢索與白名單純函式；不執行模型產生的任意程式。"""

import json
import re
from collections import Counter


def lexical_terms(text):
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def retrieve(query, documents, k=2):
    terms = Counter(lexical_terms(query))
    scored = []
    for document in documents:
        counts = Counter(lexical_terms(document["text"]))
        score = sum(min(counts[t], amount) for t, amount in terms.items())
        if score:
            scored.append((score, document))
    return [d for _, d in sorted(scored, key=lambda item: (-item[0], item[1]["id"]))[:k]]


def call_tool(request):
    if isinstance(request, str):
        request = json.loads(request)
    if set(request) != {"name", "arguments"} or request["name"] not in ("add", "multiply"):
        raise ValueError("只接受 add/multiply 與明確 arguments")
    args = request["arguments"]
    if not isinstance(args, dict) or set(args) != {"a", "b"}:
        raise ValueError("arguments 需要且只接受 a、b")
    if any(type(v) not in (int, float) for v in args.values()):
        raise ValueError("a、b 必須是數字，bool 或程式字串都不接受")
    a, b = args["a"], args["b"]
    return a + b if request["name"] == "add" else a * b


def tool_loop(requests, max_steps=3):
    trace = []
    for step, request in enumerate(requests):
        if step >= max_steps:
            return {"status": "step_limit", "trace": trace}
        if request.get("done"):
            return {"status": "done", "answer": request.get("answer"), "trace": trace}
        try:
            result = call_tool(request)
            trace.append({"request": request, "result": result})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return {"status": "invalid_request", "error": str(error), "trace": trace}
    return {"status": "needs_more_model_output", "trace": trace}
