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
