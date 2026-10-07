from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer
import json
from pathlib import Path
messages = [{"role":"user","content":"那個問題要怎麼處理？"},{"role":"assistant","content":"請說明是地址、App還是卡片的問題。"}]
tok=CharacterTokenizer.build([m["content"] for m in messages])
row={"id":"teaching-demo","task":"text","messages":messages}
enc=RecordEncoder(tok,".")
encoded=enc.encode(row)
targets=[v for v in encoded["labels"] if v!=-100]
print(tok.decode(targets,skip_special_tokens=False))
assert targets==tok.encode(messages[1]["content"])+[tok.eos_id]
print("sft targets",len(targets),"input",len(encoded["input_ids"]),"EOS",tok.eos_id)
pre=enc.encode(row,pretrain=True)
print("pretrain targets",tok.decode([v for v in pre["labels"] if v!=-100],skip_special_tokens=False))
gen=enc.encode(row,generation=True)
assert all(v==-100 for v in gen["labels"]);print("generation current target omitted",len(gen["input_ids"]))
allrows=[json.loads(s) for s in Path("outputs/p7-technical-e-cache/data/text-tools-train.jsonl").read_text().splitlines()]
needs=[messages[1]["content"],"先檢查網路並重新啟動App，仍失敗再詢問官方客服。","計算器目前不可用，請開啟後再計算。","這超出我的範圍，我能協助有限聊天、地址、App、卡片或加減乘問題。"]
for need in needs:
 rs=[r for r in allrows if r["messages"][-1]["content"]==need];assert rs
 print(json.dumps({"target":need,"count":len(rs),"original_row":rs[0]},ensure_ascii=False))
