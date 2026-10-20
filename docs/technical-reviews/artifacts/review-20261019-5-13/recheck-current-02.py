"""核對參考文件修訂、證據延續性與模態入口的有限行為。"""

import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from scripts import evaluate, train
from tiny_perceptron.data import ByteTokenizer
from tiny_perceptron.model import ModelConfig,TinyLM
from tiny_perceptron.multimodal import MultiModalLM

BASE=Path(__file__).resolve().parent
PREVIOUS="docs/course-review-20261019/reviews/rv1019_technical_5-13/technical-5-13-current-01.json"
PREVIOUS_SHA="0fe53eb3d009543bfcb8cc9278d20efe2b7b6f96042068711959824fb3ea424c"
SOURCE_SHA="b605ca3c39976c1d3b5da3efbf1ee74d7e0416c17ac6071ff65a36c59fdf6aa0"
FREEZE="docs/course-review-20261019/freeze-07.json"
FREEZE_SHA="67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
TRAINING_SHA="1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"

def sha(raw):return hashlib.sha256(raw).hexdigest()
def file_sha(path):return sha((ROOT/path).read_bytes())
def run(argv):
    result=subprocess.run(argv,cwd=ROOT,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True,timeout=90)
    return {"argv":argv,"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
def source_functions(path,names):
    text=(ROOT/path).read_text();tree=ast.parse(text)
    return {"path":path,"sha256":file_sha(path),"functions":{node.name:ast.get_source_segment(text,node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names}}

assert file_sha(PREVIOUS)==PREVIOUS_SHA
assert file_sha(FREEZE)==FREEZE_SHA
assert file_sha("course/training.md")==TRAINING_SHA
previous=json.loads((ROOT/PREVIOUS).read_bytes())
assert previous["issues"]==[]
page=next(item for item in json.loads((ROOT/FREEZE).read_bytes())["pages"] if item["page_id"]=="5.13")
assert page["source_sha256"]==SOURCE_SHA and page["figures_sha256"]=={}
assert file_sha(page["snapshot"])==SOURCE_SHA
sources=[{"id":item["id"],"path":item["path"],"previous_sha256":item["sha256"],"current_sha256":file_sha(item["path"])} for item in previous["sources"] if item["kind"]=="repository_code"]
assert [item["path"] for item in sources if item["previous_sha256"]!=item["current_sha256"]]==["course/training.md"]
artifacts=[{"id":item["id"],"path":item["path"],"previous_sha256":item["sha256"],"current_sha256":file_sha(item["path"])} for item in previous["artifacts"]]
assert all(item["previous_sha256"]==item["current_sha256"] for item in artifacts)
before=run(["git","show","ecf6dec71280:course/training.md"])
after=run(["git","show","c6b9339d2595:course/training.md"])
diff=run(["git","diff","ecf6dec71280","c6b9339d2595","--","course/training.md"])
assert before["exit_code"]==after["exit_code"]==diff["exit_code"]==0
assert sha(before["stdout"].encode())==next(item["previous_sha256"] for item in sources if item["path"]=="course/training.md")
assert sha(after["stdout"].encode())==TRAINING_SHA
reproduction=".venv/bin/python -m scripts.course_experiments.run --experiment text_foundation --device cuda"
assert reproduction in before["stdout"] and reproduction in after["stdout"]
help_result=run([sys.executable,"-m","scripts.course_experiments.run","--help"])
assert help_result["exit_code"]==0 and "--experiment" in help_result["stdout"] and "--device" in help_result["stdout"]

section=run([sys.executable,"docs/review-tools/section_facts.py","course/chapters/05.md#5.13","--execute"])
assert section["exit_code"]==0
summary=json.loads(section["stdout"])
assert summary["source_sha256"]==SOURCE_SHA and summary["svg_references"]==0
run_directory=Path(summary["artifact_directory"])
outputs={}
for name in ("extraction.json","execution.json","environment.json","stdout.txt","stderr.txt"):
    raw=(run_directory/name).read_bytes()
    outputs[name]={"sha256":sha(raw),"content":json.loads(raw) if name.endswith(".json") else raw.decode()}
assert outputs["stdout.txt"]["sha256"]==file_sha("docs/technical-reviews/artifacts/review-20261019-5-13/section-stdout.txt")
assert outputs["stderr.txt"]["content"]=="" and outputs["environment.json"]["content"]["guard_events"]==[]
parameters={str(width):TinyLM(ModelConfig(width=width)).description()["parameters"] for width in (8,16,32)}
assert parameters=={"8":6104,"16":13744,"32":33632}

class CaptureModal:
    def __init__(self):self.calls=[]
    def __call__(self,ids,labels,*,image,waveform):
        self.calls.append({"input_shape":list(ids.shape),"target_count":int((labels!=-100).sum()),"answer":ByteTokenizer().decode(labels[labels!=-100].tolist()),"image_shape":None if image is None else list(image.shape),"waveform_shape":None if waveform is None else list(waveform.shape)})
        return {"logits":torch.zeros(labels.numel(),264),"labels":labels,"auxiliary":torch.tensor(0.0)}

torch.set_num_threads(1)
synthetic=[]
for task in ("vision","audio","joint"):
    args=train.parser().parse_args(["--task",task])
    assert args.data is None and train.prepare_examples(args,ByteTokenizer()) is None
    capture=CaptureModal();random.seed(42)
    state=random.getstate()
    with patch.object(train.random,"choice",wraps=random.choice) as choices:
        losses=[float(train.modal_loss(capture,task,ByteTokenizer(),"cpu")) for _ in range(2)]
        assert choices.call_count==6
    assert random.getstate()!=state
    synthetic.append({"task":task,"data_default":args.data,"prepare_examples":None,"random_choice_calls_for_two_examples":6,"rng_state_changed":True,"captured_inputs":capture.calls,"losses":losses,"scope":"只執行例子生成與uniform-logits的loss；模型替身記錄輸入，沒有真實模型forward、backward或權重更新。"})
main_source=source_functions("scripts/train.py",{"main","prepare_examples","modal_loss"})
main_ast=ast.parse(main_source["functions"]["main"])
call_names=sorted({node.func.id if isinstance(node.func,ast.Name) else node.func.attr for node in ast.walk(main_ast) if isinstance(node,ast.Call) and isinstance(node.func,(ast.Name,ast.Attribute))})
assert not ({"evaluate","evaluate_lm","_evaluate","_nll"}&set(call_names))
assert not any("validation" in flag for action in train.parser()._actions for flag in action.option_strings)

model=MultiModalLM(TinyLM(ModelConfig(width=8)),vision_width=8,audio_width=8)
rejection=None
with patch.object(evaluate,"load_checkpoint",return_value=(model,{})),patch.object(sys,"argv",["scripts/evaluate.py","unused.pt","--device","cpu"]):
    try:evaluate.main()
    except ValueError as error:rejection=str(error)
assert rejection=="此入口只評估 TinyLM 文字 checkpoint；多模態模型請使用對應評估入口"

vqa_path="docs/course-experiments/results/vqa.json"
vqa=json.loads((ROOT/vqa_path).read_bytes())
modality_path="scripts/course_experiments/modalities.py"
assert file_sha(modality_path)==vqa["code_sha256"][modality_path]
assert vqa["evidence_status"]=="complete_run" and vqa["status"]=="completed"
paired=[]
for name,item in vqa["results"]["variants"].items():
    if "text_before" not in item:continue
    left,right=item["text_before"],item["text_after"]
    targets=lambda record:[{"messages":sample["messages"],"expected":sample["expected"]} for sample in record["samples"]]
    assert targets(left)==targets(right)
    assert left["records"]==right["records"]==10
    assert left["effective_tokens"]==right["effective_tokens"]==69
    paired.append({"variant":name,"before_records":left["records"],"after_records":right["records"],"effective_targets":69,"paired_questions_and_targets_sha256":sha(json.dumps(targets(left),ensure_ascii=False,sort_keys=True).encode()),"before_exact_match":left["exact_match"],"after_exact_match":right["exact_match"]})
assert len(paired)==4
vqa_source=source_functions(modality_path,{"run_vqa"})
assert 'text_holdout = text_splits["test"]' in vqa_source["functions"]["run_vqa"]
assert vqa_source["functions"]["run_vqa"].count('evaluate_lm(model.language, text_holdout, mode="sft")')==2
infer_source=source_functions("scripts/infer_modal.py",{"main"})
assert "evaluate_lm" not in infer_source["functions"]["main"]
code=[main_source,source_functions("scripts/evaluate.py",{"main"}),source_functions("tiny_perceptron/multimodal.py",{"MultiModalLM"}),source_functions("tiny_perceptron/training.py",{"load_checkpoint"}),infer_source,vqa_source]
changes=[
 {"statement":"三條未指定--data的本機模態路線，每步隨機產生訓練例子。","status":"verified","evidence":"train.main每步每筆呼叫modal_loss；record=None分支每次呼叫三次random.choice。兩例探測每路線均觀察六次choice。"},
 {"statement":"這三題只是抽查；這些本機路線沒有另外準備固定驗證題。","status":"verified","evidence":"三條命令沒有--data；prepare_examples為None；main沒有建驗證切分或評估呼叫；infer_modal只回單題生成。不能把pretrain_encoders的入口驗證題當作這些生成路線的固定驗證集。"},
 {"statement":"固定題比較的做法位於『重跑第11章的固定實驗』。","status":"verified","evidence":"現行摺疊區有encoders、projector與vqa指令；vqa原碼與完成紀錄保存固定切分及前後評估。"},
 {"statement":"evaluate.py只接受純文字checkpoint，多模態checkpoint會報錯。","status":"verified","evidence":"實際呼叫evaluate.main，以mock loader交回真實MultiModalLM物件，觀察預期ValueError；沒有讀取或寫出模型檔。"},
 {"statement":"上方本機命令沒有檢查文字能力是否退步。","status":"verified","evidence":"train.main未做前後文字留出評估，infer_modal只生成一題；並未把圖片或聲音回答當成文字保留成績。"},
 {"statement":"文字能力應在圖文訓練前後，以同一批文字題比較；固定vqa已做此對照。","status":"verified","evidence":"run_vqa兩次以同一text_holdout=test呼叫evaluate_lm(model.language,...)；當次code_sha256等於現行檔。四個完成變式的前後各10題、69有效目標，題目與答案完全相同。未重訓或重新測量loss。"},
]
continuity=[{"claim_id":claim["id"],"status":"verified","reason":"小節原文未變；除training.md外所有repo來源與原artifact SHA皆未變。" if claim["id"]!="c41" else "重新讀取兩端與現行training.md，重做text_foundation命令不變，CLI help再次成功。"} for claim in previous["claims"]]
receipt={"command":"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-13/recheck-current-02.py","environment":{"python":platform.python_version(),"torch":str(torch.__version__),"device":"cpu","platform":platform.platform()},"previous_report":{"path":PREVIOUS,"sha256":PREVIOUS_SHA},"source_freeze":{"path":FREEZE,"sha256":FREEZE_SHA},"source_sha256":SOURCE_SHA,"figure_sha256":{},"previous_issues":previous["issues"],"resolved_issue_ids":[],"source_hash_checks":sources,"artifact_hash_checks":artifacts,"training_diff":diff,"training_versions":{"before":"ecf6dec71280","after":"c6b9339d2595","before_sha256":sha(before["stdout"].encode()),"after_sha256":sha(after["stdout"].encode())},"reproduction_command":{"text":reproduction,"present_before_and_current":True,"help":help_result},"section_execution":{"invocation":section,"summary":summary,"copied_outputs":outputs},"parameters":parameters,"synthetic_examples_probe":synthetic,"local_training_main_call_names":call_names,"local_training_fixed_validation_available":False,"multimodal_evaluation_rejection":{"model_type":type(model).__name__,"message":rejection,"scope":"只替換checkpoint讀取來源，實際執行現行evaluate.main的型別守門；未用真實checkpoint。"},"vqa_record":{"path":vqa_path,"sha256":file_sha(vqa_path),"evidence_status":vqa["evidence_status"],"code_sha256_matches_current":True,"text_data":vqa["results"]["text_data"],"paired_variants":paired},"code_receipts":code,"changed_sentence_checks":changes,"claim_continuity":continuity,"previous_dependency_sentence_superseded":"上一份對『再拿整份驗證題』作為一般診斷建議的判斷，不再用於現行文字。實際確認這三條本機路線沒有另備固定驗證題；新文改以抽查及獨立固定實驗區分。該變動沒有影響本節text_foundation重做入口。","result":"全部頁面主張維持核實；參考文件新句符合現行程式與固定實驗紀錄。","limitations":["沒有重跑完整訓練、GPU/MPS運算或checkpoint的實際能力評估。","固定vqa完成結果依既有原始紀錄查核，未獨立重訓；本次模態探測用模型替身，型別拒絕探測用mock loader。","不宣稱重審整份training.md或做初學者理解審閱。"]}
with (BASE/"recheck-current-02.json").open("x",encoding="utf-8") as handle:handle.write(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"result":receipt["result"],"claims":len(continuity),"prior_issues":len(previous["issues"]),"changed_sentence_checks":len(changes),"section_exit_code":section["exit_code"],"fixed_text_comparison_variants":len(paired)},ensure_ascii=False))
