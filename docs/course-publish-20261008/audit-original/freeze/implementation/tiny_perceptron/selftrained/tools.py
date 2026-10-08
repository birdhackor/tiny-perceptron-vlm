"""A bounded calculator and one-hop model/tool/model protocol.

Only a model-generated JSON object can request arithmetic. The runtime returns
structured values, never a question-dependent Chinese answer. This module has
no learned weights, shell execution, expression interpreter, or answer table.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

OPERATIONS = ("add", "subtract", "multiply")
MAX_OPERAND = 99
MAX_CALL_CHARACTERS = 256
CALL_KEYS = frozenset(("tool", "operation", "a", "b"))
PUBLIC_MODAL_KEYS = frozenset(("image", "audio", "roi", "image_layout"))
TOOLS_SCHEMA = (
    "計算器可用；整數參數為0到99。計算時輸出JSON："
    '{"tool":"calculator","operation":"add|subtract|multiply","a":整數,"b":整數}。'
    "收到tool結果後再回答；缺參數先詢問；不需計算時直接回答。"
)


class ToolCallError(ValueError):
    """The output is not an exact, permitted calculator call."""


@dataclass(frozen=True)
class CalculatorCall:
    operation: str
    a: int
    b: int

    def to_dict(self) -> dict:
        return {"tool": "calculator", "operation": self.operation, "a": self.a, "b": self.b}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ToolCallError("duplicate JSON key")
        value[key] = item
    return value


def _validate_call(call: CalculatorCall) -> CalculatorCall:
    if call.operation not in OPERATIONS:
        raise ToolCallError("operation is not permitted")
    for operand in (call.a, call.b):
        # bool is an int subclass, and must not become an arithmetic parameter.
        if type(operand) is not int or not 0 <= operand <= MAX_OPERAND:
            raise ToolCallError("operands must be integers from 0 to 99")
    return call


def parse_tool_call(text: str) -> CalculatorCall:
    """Require exactly four unique JSON keys and bounded integer operands."""
    if not isinstance(text, str) or len(text) > MAX_CALL_CHARACTERS:
        raise ToolCallError("call must be a bounded string")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ToolCallError("invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != CALL_KEYS:
        raise ToolCallError("call must contain exactly tool, operation, a, b")
    if value["tool"] != "calculator":
        raise ToolCallError("tool is not permitted")
    return _validate_call(CalculatorCall(value["operation"], value["a"], value["b"]))


def serialize_tool_call(operation: str, a: int, b: int) -> str:
    return json.dumps(_validate_call(CalculatorCall(operation, a, b)).to_dict(), separators=(",", ":"))


def execute_tool_call(call: CalculatorCall | str) -> dict:
    """Execute arithmetic itself; operands alone determine the integer result."""
    if isinstance(call, str):
        call = parse_tool_call(call)
    if not isinstance(call, CalculatorCall):
        raise ToolCallError("executor requires a validated calculator call")
    call = _validate_call(call)
    if call.operation == "add":
        result = call.a + call.b
    elif call.operation == "subtract":
        result = call.a - call.b
    else:
        result = call.a * call.b
    return {"tool": "calculator", "ok": True, "result": result}


def serialize_tool_result(result: Mapping) -> str:
    """Serialize real results or explicit test-only replacement values."""
    if result.get("tool") != "calculator" or type(result.get("ok")) is not bool:
        raise ToolCallError("invalid tool result")
    if result["ok"]:
        if set(result) != {"tool", "ok", "result"} or type(result["result"]) is not int:
            raise ToolCallError("a successful result needs exactly one integer value")
        if not -MAX_OPERAND <= result["result"] <= MAX_OPERAND**2:
            raise ToolCallError("tool result is outside its representable range")
    elif set(result) != {"tool", "ok", "error"} or result["error"] not in ("invalid_call", "unavailable"):
        raise ToolCallError("invalid tool error")
    return json.dumps(dict(result), separators=(",", ":"))


def _messages_only(messages: Sequence[Mapping]) -> list[dict[str, object]]:
    copied = []
    for message in messages:
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("unknown message role")
        if not isinstance(message.get("content"), str):
            raise ValueError("message content must be text")
        public = {"role": message["role"], "content": message["content"]}
        # These values are loader metadata, not text. Preserve them so a tool
        # hop cannot silently remove actual image/audio context. Gold labels
        # and other unknown keys never pass this public-message boundary.
        public.update({key: message[key] for key in PUBLIC_MODAL_KEYS if key in message})
        copied.append(public)
    return copied


def run_tool_loop(
    generate: Callable[[list[dict[str, object]]], str],
    messages: Sequence[Mapping],
    *,
    tools_enabled: bool = True,
) -> dict:
    """Run at most one real calculator hop, then use the same generator again.

    The caller supplies prompt messages, without an assistant target. A JSON-like
    model output requests a tool; plain text is a direct reply. Malformed JSON is
    a recorded validation failure, not executable code. Error handling also uses
    a second model generation; the runtime never writes the final answer.
    """
    prompt = _messages_only(messages)
    initial = generate(prompt)
    if not isinstance(initial, str):
        raise TypeError("generator output must be text")
    trace = {
        "initial_output": initial,
        "tool_call": None,
        "tool_result": None,
        "tool_message": None,
        "final_output": initial,
        "executed": False,
        "status": "no_tool",
    }
    if not initial.lstrip().startswith(("{", "[", "```")):
        return trace
    try:
        call = parse_tool_call(initial)
        trace["tool_call"] = call.to_dict()
    except ToolCallError:
        result = {"tool": "calculator", "ok": False, "error": "invalid_call"}
        trace["status"] = "invalid_call"
    else:
        if not tools_enabled:
            result = {"tool": "calculator", "ok": False, "error": "unavailable"}
            trace["status"] = "unavailable"
        else:
            result = execute_tool_call(call)
            trace["executed"] = True
            trace["status"] = "executed"
    tool_message = {"role": "tool", "content": serialize_tool_result(result)}
    trace["tool_result"] = result
    trace["tool_message"] = tool_message
    trace["final_output"] = generate(prompt + [{"role": "assistant", "content": initial}, tool_message])
    if not isinstance(trace["final_output"], str):
        raise TypeError("generator output must be text")
    return trace


# Small compatibility aliases; all use the same strict implementation.
parse_call = parse_tool_call
execute = execute_tool_call
