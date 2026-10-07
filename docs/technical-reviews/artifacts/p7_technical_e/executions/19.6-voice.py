import json,torch
from pathlib import Path
from tiny_perceptron.selftrained.inference import InferenceAssistant,attach_public_metadata
torch.set_num_threads(2)
a=InferenceAssistant("outputs/p7-technical-e-cache/moe-joint","outputs/p7-technical-e-cache/data",manifest_sha256="f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e")
msgs=json.loads(Path("docs/selftrained/examples/v2/voice_qa.messages.json").read_text())
audio="audio/ef25d9a7a3a6ce3790a040ba.wav"
v=a.reply(msgs,task="voice_qa",audio=audio,modality_message_index=3,max_new_tokens=128)
p=Path("docs/technical-reviews/artifacts/p7_technical_e/executions/19.6-voice-result.json");assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+"\n")
history=attach_public_metadata(msgs,audio=audio,modality_message_index=3)
row={"id":"public-head-check","task":"voice_qa","messages":history}
e=a.encoder.encode(row,generation=True,messages=history);payload=e["modalities"][0]
with torch.no_grad():tokens,logits=a.model.audio(payload["values"],payload["valid"])
raw=[json.loads(s) for s in Path("outputs/p7-technical-e-cache/data/voice-validation.jsonl").read_text().splitlines()]
matches=[r for r in raw if r.get("audio")==audio];assert matches
gold=matches[0]["supervision"]["intent_id"]
print(json.dumps({"answer":v["answer"],"eos":v["generations"][0]["eos"],"audio":audio,"frames_shape":list(payload["values"].shape),"tokens_shape":list(tokens.shape),"head_logits":logits.tolist(),"head_probs":logits.softmax(-1).tolist(),"head_argmax":int(logits.argmax()),"gold_intent":gold,"original_record_id":matches[0]["id"],"prompt_messages":v["generations"][0]["prompt_messages"]},ensure_ascii=False))
assert int(logits.argmax())!=gold
print("classification_head_incorrect_independent_of_generated_answer")
