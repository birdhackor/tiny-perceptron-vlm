tiny_perceptron/selftrained/tools.py:L47-L103
47: def _validate_call(call: CalculatorCall) -> CalculatorCall:
48:     if call.operation not in OPERATIONS:
49:         raise ToolCallError("operation is not permitted")
50:     for operand in (call.a, call.b):
51:         # bool is an int subclass, and must not become an arithmetic parameter.
52:         if type(operand) is not int or not 0 <= operand <= MAX_OPERAND:
53:             raise ToolCallError("operands must be integers from 0 to 99")
54:     return call
55: 
56: 
57: def parse_tool_call(text: str) -> CalculatorCall:
58:     """Require exactly four unique JSON keys and bounded integer operands."""
59:     if not isinstance(text, str) or len(text) > MAX_CALL_CHARACTERS:
60:         raise ToolCallError("call must be a bounded string")
61:     try:
62:         value = json.loads(text, object_pairs_hook=_unique_object)
63:     except (json.JSONDecodeError, RecursionError) as exc:
64:         raise ToolCallError("invalid JSON") from exc
65:     if not isinstance(value, dict) or set(value) != CALL_KEYS:
66:         raise ToolCallError("call must contain exactly tool, operation, a, b")
67:     if value["tool"] != "calculator":
68:         raise ToolCallError("tool is not permitted")
69:     return _validate_call(CalculatorCall(value["operation"], value["a"], value["b"]))
70: 
71: 
72: def serialize_tool_call(operation: str, a: int, b: int) -> str:
73:     return json.dumps(_validate_call(CalculatorCall(operation, a, b)).to_dict(), separators=(",", ":"))
74: 
75: 
76: def execute_tool_call(call: CalculatorCall | str) -> dict:
77:     """Execute arithmetic itself; operands alone determine the integer result."""
78:     if isinstance(call, str):
79:         call = parse_tool_call(call)
80:     if not isinstance(call, CalculatorCall):
81:         raise ToolCallError("executor requires a validated calculator call")
82:     call = _validate_call(call)
83:     if call.operation == "add":
84:         result = call.a + call.b
85:     elif call.operation == "subtract":
86:         result = call.a - call.b
87:     else:
88:         result = call.a * call.b
89:     return {"tool": "calculator", "ok": True, "result": result}
90: 
91: 
92: def serialize_tool_result(result: Mapping) -> str:
93:     """Serialize real results or explicit test-only replacement values."""
94:     if result.get("tool") != "calculator" or type(result.get("ok")) is not bool:
95:         raise ToolCallError("invalid tool result")
96:     if result["ok"]:
97:         if set(result) != {"tool", "ok", "result"} or type(result["result"]) is not int:
98:             raise ToolCallError("a successful result needs exactly one integer value")
99:         if not -MAX_OPERAND <= result["result"] <= MAX_OPERAND**2:
100:             raise ToolCallError("tool result is outside its representable range")
101:     elif set(result) != {"tool", "ok", "error"} or result["error"] not in ("invalid_call", "unavailable"):
102:         raise ToolCallError("invalid tool error")
103:     return json.dumps(dict(result), separators=(",", ":"))

tiny_perceptron/selftrained/tools.py:L122-L170
122: def run_tool_loop(
123:     generate: Callable[[list[dict[str, object]]], str],
124:     messages: Sequence[Mapping],
125:     *,
126:     tools_enabled: bool = True,
127: ) -> dict:
128:     """Run at most one real calculator hop, then use the same generator again.
129: 
130:     The caller supplies prompt messages, without an assistant target. A JSON-like
131:     model output requests a tool; plain text is a direct reply. Malformed JSON is
132:     a recorded validation failure, not executable code. Error handling also uses
133:     a second model generation; the runtime never writes the final answer.
134:     """
135:     prompt = _messages_only(messages)
136:     initial = generate(prompt)
137:     if not isinstance(initial, str):
138:         raise TypeError("generator output must be text")
139:     trace = {
140:         "initial_output": initial,
141:         "tool_call": None,
142:         "tool_result": None,
143:         "tool_message": None,
144:         "final_output": initial,
145:         "executed": False,
146:         "status": "no_tool",
147:     }
148:     if not initial.lstrip().startswith(("{", "[", "```")):
149:         return trace
150:     try:
151:         call = parse_tool_call(initial)
152:         trace["tool_call"] = call.to_dict()
153:     except ToolCallError:
154:         result = {"tool": "calculator", "ok": False, "error": "invalid_call"}
155:         trace["status"] = "invalid_call"
156:     else:
157:         if not tools_enabled:
158:             result = {"tool": "calculator", "ok": False, "error": "unavailable"}
159:             trace["status"] = "unavailable"
160:         else:
161:             result = execute_tool_call(call)
162:             trace["executed"] = True
163:             trace["status"] = "executed"
164:     tool_message = {"role": "tool", "content": serialize_tool_result(result)}
165:     trace["tool_result"] = result
166:     trace["tool_message"] = tool_message
167:     trace["final_output"] = generate(prompt + [{"role": "assistant", "content": initial}, tool_message])
168:     if not isinstance(trace["final_output"], str):
169:         raise TypeError("generator output must be text")
170:     return trace
