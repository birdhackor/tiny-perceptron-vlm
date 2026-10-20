"""核對引用文件的音訊規約與既有證據是否仍適用。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tiny_perceptron.modal_data import modal_example
from tiny_perceptron.multimodal import tone


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv):
    result=subprocess.run(argv,cwd=ROOT,capture_output=True,check=False)
    return {"argv":argv,"exit_code":result.returncode,"stdout":result.stdout.decode(),"stderr":result.stderr.decode()}


def audio_paragraph(text):
    return next(p for p in text.split("\n\n") if p.startswith("自己的圖像紀錄可寫成"))


previous_path=ROOT/"docs/course-review-20261019/reviews/rv1019_technical_12-2/technical-12-2-current-01.json"
previous=json.loads(previous_path.read_bytes())
source_path=ROOT/"docs/course-review-20261019/freeze-05/sources/12.2.md"
freeze_path=ROOT/"docs/course-review-20261019/freeze-07.json"
source=source_path.read_bytes()
live=(ROOT/"course/chapters/12.md").read_bytes()
section=re.search(rb"(?ms)^## 12\.2 .*?(?=^## |\Z)",live)[0]
result={
    "command":"PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-12-2/verify-current-02.py",
    "environment":{"python":platform.python_version(),"torch":str(torch.__version__),"numpy":str(np.__version__),"device":"cpu","platform":platform.platform()},
    "checked_at_utc":datetime.now(UTC).isoformat(),
    "previous_report":{"path":str(previous_path.relative_to(ROOT)),"sha256":sha(previous_path)},
    "original_report":{"path":previous["original_report"]["path"],"sha256":sha(ROOT/previous["original_report"]["path"])},
    "source":{"path":str(source_path.relative_to(ROOT)),"sha256":sha(source_path),"unchanged_since_previous":sha(source_path)==previous["source_sha256"],"live_matches_freeze":section==source,"freeze_manifest_sha256":sha(freeze_path)},
    "dependency_diff":command(["git","diff","ecf6dec71280","c6b9339d2595","--","course/training.md"]),
    "repository_revision":command(["git","rev-parse","HEAD"])["stdout"].strip(),
}
assert result["previous_report"]["sha256"]=="98a224362203fb6e7e41620e915b9db835bd13bd10399685736820bb2fea5187"
assert result["original_report"]["sha256"]=="c5d2a80aa7e81a5cd42fdab9829652998e512aaf82d4f58075915c0ef661413d"
assert result["source"]["sha256"]=="05eb1ea9b9088d0f2496f59613bdbb461133c76e9a05819f509d1c6c63162945"
assert result["source"]["live_matches_freeze"]
assert result["source"]["freeze_manifest_sha256"]=="67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
result["inherited_evidence"]=[]
for group in ("sources","artifacts"):
    for item in previous[group]:
        if "path" not in item:
            continue
        actual=sha(ROOT/item["path"])
        result["inherited_evidence"].append({"group":group,"id":item["id"],"path":item["path"],"previous_sha256":item["sha256"],"current_sha256":actual,"unchanged":actual==item["sha256"]})
        if item["id"]!="training_doc":
            assert actual==item["sha256"],item["path"]
old_training=command(["git","show","ecf6dec71280:course/training.md"])
assert old_training["exit_code"]==0
training=(ROOT/"course/training.md").read_text()
result["dependency_audio_scope"]={
    "path":"course/training.md","previous_version_full_sha256":hashlib.sha256(old_training["stdout"].encode()).hexdigest(),
    "current_sha256":sha(ROOT/"course/training.md"),"quoted_paragraph":audio_paragraph(training),
    "relevant_paragraph_unchanged":audio_paragraph(old_training["stdout"])==audio_paragraph(training),
    "review_scope":"本節只依此文件支持非空、單聲道、16000 Hz 的音訊格式限制；其他修改段落不作為本節主張的依據。",
}
assert result["dependency_audio_scope"]["previous_version_full_sha256"]=="d59ac50bedd0cf55a5f8cc7dc99bee50799750a7c31c101ae0b528dc3a5b1e94"
assert result["dependency_audio_scope"]["current_sha256"]=="1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
assert result["dependency_audio_scope"]["relevant_paragraph_unchanged"]

run_path=ROOT/"outputs/reviewer-tools/runs/20261020T151502506607Z"
result["section_run"]={"command":'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/12.md#12.2" --execute',"copied_from":str(run_path.relative_to(ROOT)),"files":{}}
for name in ("stdout.txt","stderr.txt","environment.json","execution.json","extraction.json"):
    raw=(run_path/name).read_bytes()
    result["section_run"]["files"][name]={"sha256":hashlib.sha256(raw).hexdigest(),"content":json.loads(raw) if name.endswith(".json") else raw.decode()}
assert (run_path/"section.md").read_bytes()==source
assert result["section_run"]["files"]["execution.json"]["content"]["exit_code"]==0
result["input_validation"]=[]
for name,values,rate,should_accept in (
    ("有效單聲道",np.zeros(16,dtype=np.float32),16000,True),
    ("錯誤取樣率",np.zeros(16,dtype=np.float32),8000,False),
    ("空錄音",np.zeros(0,dtype=np.float32),16000,False),
    ("多聲道",np.zeros((16,2),dtype=np.float32),16000,False)):
    with patch("tiny_perceptron.modal_data.sf.read",return_value=(values,rate)):
        try:
            modal_example({"audio":"voice.wav","question":"digit?","answer":"0"},ROOT,"audio")
            accepted=True
            observed="接受"
        except ValueError as error:
            accepted=False
            observed=str(error)
    result["input_validation"].append({"case":name,"shape":list(values.shape),"rate":rate,"accepted":accepted,"expected_acceptance":should_accept,"observed":observed})
    assert accepted==should_accept,name
result["alias_checks"]=[]
for high_hz,low_hz in ((9000.0,7000.0),(9500.0,6500.0)):
    high=tone(high_hz,seconds=0.001)
    low=tone(low_hz,seconds=0.001)
    other=-low
    explicit=bool(((high-other).abs()<=1e-5+1e-5*other.abs()).all())
    observed=torch.allclose(high,other,atol=1e-5)
    result["alias_checks"].append({"frequencies_hz":[high_hz,low_hz],"samples":len(high),"dtype":str(high.dtype),"max_absolute_difference":float((high-other).abs().max()),"formula_all":explicit,"allclose":observed})
    assert explicit==observed and observed
result["arithmetic"]={"duration":[1600/16000,1600/8000],"samples_per_cycle":[round(16000/440,2),round(8000/440,2)],"nyquist":[16000/2,8000/2],"relabeled_frequency":440*8000/16000,"recording_duration":[4591/8000,9182/16000],"exercise_samples":8000*16000//8000}
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
