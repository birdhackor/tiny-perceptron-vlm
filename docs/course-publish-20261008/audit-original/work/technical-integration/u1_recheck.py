import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import soundfile as sf
from tiny_perceptron.selftrained.inference import InferenceAssistant, attach_public_metadata

ROOT=Path('/workspace/work/tutorial-audit-20261008')
REPO=Path('/workspace/tiny-perceptron-vlm')
WORK=ROOT/'work/technical-integration'
MODEL=Path('/workspace/selftrained-v2/outputs/p6-T.11-public/moe-joint')
DATA=Path('/workspace/selftrained-v2/outputs/selftrained-v2/data')
AUDIO='audio/ef25d9a7a3a6ce3790a040ba.wav'
EXPECTED_MODEL='26102d2edbd4574bef6733cba9e0dfd039118b206babdc66dbd16d1e57032d8c'
EXPECTED_MANIFEST='f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e'
INITIAL=ROOT/'reports/technical-integration-initial.json'
SEAL=ROOT/'reports/technical-integration-initial.seal.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

initial_before=sha(INITIAL)
seal_before=sha(SEAL)
assert initial_before==json.loads(SEAL.read_text())['report_sha256']
assert sha(MODEL/'model.safetensors')==EXPECTED_MODEL
assert sha(MODEL/'inference-manifest.json')==EXPECTED_MANIFEST
frozen=json.loads((REPO/'docs/selftrained/results/public-raw/moe/freeze/frozen.json').read_text())
# The asset map is retained in the frozen raw export; do not infer from file name.
asset_maps=[v for v in frozen.values() if isinstance(v,dict) and AUDIO in v]
assert len(asset_maps)==1
expected_audio=asset_maps[0][AUDIO]
assert sha(DATA/AUDIO)==expected_audio
reference_records=[]
for line in (REPO/'outputs/selftrained/data/voice-validation.jsonl').read_text().splitlines():
    row=json.loads(line)
    if row.get('audio')==AUDIO and row['task']=='voice_qa':
        reference_records.append(row)
assert reference_records
gold_ids={r['supervision']['intent_id'] for r in reference_records}
gold_names={r['supervision']['intent'] for r in reference_records}
assert gold_ids=={1} and gold_names=={'app_error'}
messages_path=REPO/'docs/selftrained/examples/v2/voice_qa.messages.json'
messages=json.loads(messages_path.read_text())
assert all(set(m)<= {'role','content'} for m in messages)

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
start=time.monotonic()
assistant=InferenceAssistant(MODEL, DATA, device='cpu', manifest_sha256=EXPECTED_MANIFEST)
public_history=attach_public_metadata(messages,audio=AUDIO,modality_message_index=3)
encoded=assistant.encoder.encode({'id':'u1-actual-inference','task':'voice_qa','messages':public_history},generation=True,messages=public_history)
assert len(encoded['modalities'])==1
payload=assistant.encoder.to_device(encoded['modalities'])[0]
assert payload['kind']=='audio'
with torch.no_grad():
    _,logits=assistant.model.audio_encoder(payload['values'],payload['valid'])
    head_logits=logits.tolist()
    head_probs=logits.softmax(-1).tolist()
    head_predicted_id=int(logits.argmax())
result=assistant.reply(messages,task='voice_qa',max_new_tokens=128,audio=AUDIO,modality_message_index=3)
seconds=time.monotonic()-start
saved=json.loads((REPO/'docs/selftrained/results/public-cpu-raw/voice_qa-1-result.json').read_text())
waveform,sample_rate=sf.read(DATA/AUDIO,dtype='float32')
raw={
 'schema_version':1,'issue':'U1','scope':'One existing published-weight CPU audio-head and first-answer recheck; no training or large mature models.',
 'runtime':{'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'device':'cpu','seconds':seconds},
 'source_files_sha256':{name:sha(ROOT/'freeze/implementation'/name) for name in ['tiny_perceptron/selftrained/model.py','tiny_perceptron/selftrained/dataset.py','tiny_perceptron/selftrained/inference.py']},
 'existing_model_dir':str(MODEL),'model_files':{name:{'sha256':sha(MODEL/name),'bytes':(MODEL/name).stat().st_size} for name in ['model.safetensors','model-config.json','tokenizer.json','inference-manifest.json']},
 'audio':{'path':str(DATA/AUDIO),'sha256':sha(DATA/AUDIO),'bytes':(DATA/AUDIO).stat().st_size,'sample_rate':sample_rate,'samples':len(waveform),'seconds':len(waveform)/sample_rate,'preprocessed_shape':list(payload['values'].shape)},
 'reference':{'source':str(REPO/'outputs/selftrained/data/voice-validation.jsonl'),'source_sha256':sha(REPO/'outputs/selftrained/data/voice-validation.jsonl'),'record_ids':[r['id'] for r in reference_records],'intent_id':1,'intent':'app_error','used_for_comparison_only':True},
 'input':{'path':str(messages_path),'sha256':sha(messages_path),'messages':public_history,'gold_or_current_answer_in_prompt':False},
 'head':{'logits':head_logits,'probabilities':head_probs,'argmax_id':head_predicted_id,'reference_id':1,'correct':head_predicted_id==1},
 'generation':{'answer':result['answer'],'generations':result['generations'],'tool_trace':result['tool_trace'],'model_sha256':result['model']['files']['model.safetensors']},
 'comparison_to_saved_public_cpu':{'source':str(REPO/'docs/selftrained/results/public-cpu-raw/voice_qa-1-result.json'),'source_sha256':sha(REPO/'docs/selftrained/results/public-cpu-raw/voice_qa-1-result.json'),'saved_answer':saved['actual_answer'],'answer_equal':result['answer']==saved['actual_answer'],'same_weight_sha256':result['model']['files']['model.safetensors']==saved['model']['files']['model.safetensors']},
 'preservation':{'initial_sha_before':initial_before,'initial_sha_after':sha(INITIAL),'seal_sha_before':seal_before,'seal_sha_after':sha(SEAL)},
 'limits':['One known validation recording, not new heldout success.','Runtime torch2.14.1+cpu differs from original torch2.8 runtime; stable class argmax and exact saved answer checked here.','No actual audio listening by human; generation appropriateness compared to source intent and reference response.','No continuation or full test-set run.'],
}
assert raw['preservation']['initial_sha_before']==raw['preservation']['initial_sha_after']
assert raw['preservation']['seal_sha_before']==raw['preservation']['seal_sha_after']
out=WORK/'u1-head-generation-actual.json'
assert not out.exists()
out.write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(out),'head':raw['head'],'generation_answer':result['answer'],'eos':result['generations'][0]['eos'],'saved_equal':raw['comparison_to_saved_public_cpu']['answer_equal'],'seconds':seconds},ensure_ascii=False))
