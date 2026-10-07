import json,torch
from pathlib import Path
from tiny_perceptron.selftrained.inference import InferenceAssistant
torch.set_num_threads(2)
first=json.loads(Path("docs/technical-reviews/artifacts/p7_technical_e/executions/19.6-voice-corrected-result.json").read_text())
messages=first["messages"]+[{"role":"user","content":"請將剛才的答覆改成兩點。"}]
a=InferenceAssistant("outputs/p7-technical-e-cache/moe-joint","outputs/p7-technical-e-cache/data",manifest_sha256="f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e")
v=a.reply(messages,task="voice_topic_continuation",max_new_tokens=128)
p=Path("docs/technical-reviews/artifacts/p7_technical_e/executions/19.6-continuation-result.json");assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
expected="1. 先檢查網路並重新啟動App。\n2. 仍失敗再詢問官方客服。"
assert v["answer"]==expected;assert v["generations"][0]["prompt_messages"]==messages
print(json.dumps({"answer":v["answer"],"eos":v["generations"][0]["eos"],"retained_true_first_answer":messages[4],"original_audio":messages[3]["audio"],"current_user":messages[-1],"model_sha":v["generations"][0]["model_sha256"],"generated_ids":v["generations"][0]["generated_ids"]},ensure_ascii=False))
# Independently check embedded original image bytes.
import xml.etree.ElementTree as ET,base64,hashlib
svgs=["97ad8fbf6e892ccf0ca0957adcdea3dcaa78f2a252daf69147c14241aeb93d09","3e3e0469293027f8126deda55d4eaced3f9db53f6f7fdf7adcf1ed6a2bf1b53a"]
originals=[Path("outputs/p7-technical-e-cache/data")/s for s in ["images/vision/c98cb897d639c880fde261eafc6d953d.png","images/vision/13913fb756b87b5c3e0f52e4ba194fec.png","images/ocr/3a8d1184bb954b6f30c6fc09b31c678e.png"]]
expectedhash=[hashlib.sha256(p.read_bytes()).hexdigest() for p in originals];got=[]
for s in svgs:
 tree=ET.parse("docs/course-revision-20261007-phase7/reviews/freeze-03/figures/"+s+".svg")
 for e in tree.iter():
  if e.tag.endswith("}image"):
   data=next(v for k,v in e.attrib.items() if k.endswith("href"));raw=base64.b64decode(data.split(",",1)[1]);got.append(hashlib.sha256(raw).hexdigest())
assert got==expectedhash;print("embedded3PNG_match_original_bytes",got)
