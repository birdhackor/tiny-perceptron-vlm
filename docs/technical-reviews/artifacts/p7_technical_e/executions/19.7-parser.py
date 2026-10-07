from tiny_perceptron.selftrained.tools import parse_tool_call,execute_tool_call,ToolCallError,run_tool_loop
import json
raw='{"tool":"calculator","operation":"multiply","a":26,"b":16}'
call=parse_tool_call(raw)
print("合法請求",call.to_dict());print("真執行結果",execute_tool_call(call))
assert execute_tool_call(raw)["result"]==416
variant=raw.replace('"a":26','"a":25')
assert execute_tool_call(variant)["result"]==400;print("legal wrong operand",execute_tool_call(variant))
for text in [raw.replace('"a":26','"a":true'),raw.replace('"a":26','"a":100'),raw.replace('"multiply"','"divide"'),raw[:-1]+',"a":25}',raw[:-1]+',"extra":1}']:
 try:parse_tool_call(text)
 except ToolCallError as e:print("expected rejection",str(e))
 else:raise AssertionError("invalid accepted")
calls=[]
def artificial_generator(messages):
 calls.append(messages);return raw if len(calls)==1 else "故意手工錯結果。"
v=run_tool_loop(artificial_generator,[{"role":"user","content":"test"}])
assert v["tool_result"]["result"]==416 and v["final_output"]=="故意手工錯結果。"
print("artificial protocol only",v)
