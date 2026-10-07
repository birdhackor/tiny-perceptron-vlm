import json,torch
from pathlib import Path
from tiny_perceptron.selftrained.inference import InferenceAssistant
torch.set_num_threads(2)
a=InferenceAssistant("outputs/p7-technical-e-cache/moe-joint","outputs/p7-technical-e-cache/data",manifest_sha256="f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e")
msgs=json.loads(Path("docs/selftrained/examples/v2/tool_call.messages.json").read_text())
v=a.reply(msgs,task="tool_call",tools=True,max_new_tokens=128)
p=Path("docs/technical-reviews/artifacts/p7_technical_e/executions/19.7-result.json");assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
assert v["tool_trace"]["tool_call"]=={"tool":"calculator","operation":"multiply","a":26,"b":16}
assert v["tool_trace"]["tool_result"]=={"tool":"calculator","ok":True,"result":416}
assert v["answer"]=="結果是416。";assert len(v["generations"])==2
assert v["generations"][1]["prompt_messages"]==msgs+[{"role":"assistant","content":v["tool_trace"]["initial_output"]},v["tool_trace"]["tool_message"]]
assert len({g["model_sha256"] for g in v["generations"]})==1
print(json.dumps({"tool_trace":v["tool_trace"],"generations":[{"call_index":g["call_index"],"raw_output":g["raw_output"],"eos":g["eos"],"model_sha256":g["model_sha256"]} for g in v["generations"]]},ensure_ascii=False))
