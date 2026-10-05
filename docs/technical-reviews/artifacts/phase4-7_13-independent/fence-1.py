import json

reply = '{"answer": 3}'
parsed = json.loads(reply)
scores = {
    "格式符合": isinstance(parsed, dict) and set(parsed) == {"answer"} and type(parsed["answer"]) is int,
    "1+1內容正確": parsed.get("answer") == 2,
}
print(scores)
assert scores["格式符合"] and not scores["1+1內容正確"]
