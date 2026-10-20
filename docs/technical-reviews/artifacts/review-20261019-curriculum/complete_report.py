"""保留既有證據，以新增收據補齊來源定位，再產生尚未存檔的正式報告。"""

import concurrent.futures
import datetime
import hashlib
import html
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-curriculum"
COMMAND = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-curriculum/complete_report.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


preserved = {p.name: sha(p.read_bytes()) for p in BASE.iterdir() if p.is_file() and p.name != "complete_report.py"}
paths = ["docs/course-experiments/plan.json", "docs/course-experiments/results/flash_probe.json"]
local = {}
for path in paths:
    raw = (ROOT/path).read_bytes()
    value = json.loads(raw)
    if path.endswith("plan.json"):
        selection = {"sequence_count":len(value["sequence"]), "supporting_experiments":value["supporting_experiments"]}
    else:
        selection = {k:value[k] for k in ("experiment_id", "title", "device", "gpu", "status", "evidence_status", "results") if k in value}
    local[path] = {"sha256":sha(raw), "selected":selection}

jobs = [
    ("minimind-lora-examples", "https://raw.githubusercontent.com/jingyaogong/minimind/f659b55761b754d306bd140573493a6543cafd7f/README.md", ["lora_identity", "lora_medical", "私有领域"], "f659b55761b754d306bd140573493a6543cafd7f"),
    ("learner-lr-diary", "https://raw.githubusercontent.com/donglinkang2021/cs336-assignment1-basics/c7bd4a373c1c56fc6cb02cc3f848c13d02a2b825/docs/CHANGELOG.md", ["学习率", "learning rate", "lr", "Muon"], "c7bd4a373c1c56fc6cb02cc3f848c13d02a2b825"),
    ("learner-gradient-question", "https://github.com/karpathy/nanochat/discussions/856", ["backward", "optimizer", "four layers", "4 layers"], "GitHub Discussion HTML：本回合讀取"),
]


def fetch(job):
    identifier, url, needles, version = job
    command = ["curl", "-fsSL", "--max-time", "40", url]
    result = subprocess.run(command, capture_output=True)
    raw = result.stdout
    record = {"id":identifier, "url":url, "version":version, "accessed_on":datetime.date.today().isoformat(), "command":command, "exit_code":result.returncode, "sha256":sha(raw), "bytes":len(raw)}
    if result.returncode:
        record["stderr"] = result.stderr.decode(errors="replace")
        return record
    text = raw.decode(errors="replace")
    if identifier == "learner-gradient-question":
        text = re.sub(r"<script\b.*?</script>", "", text, flags=re.S|re.I)
        text = html.unescape(re.sub(r"<[^>]+>", "\n", text))
    lines = text.splitlines()
    selected = set()
    for needle in needles:
        for i in [i for i,line in enumerate(lines) if needle.lower() in line.lower()][:5]:
            selected.update(range(max(0,i-1), min(len(lines),i+2)))
    record["excerpts"] = [{"line":i+1,"text":lines[i][:1600]} for i in sorted(selected)[:45]]
    return record


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    external = list(pool.map(fetch, jobs))
supplement = {"command":COMMAND,"environment":{"python":sys.version,"platform":platform.platform(),"device":"cpu"},"preserved_inputs":preserved,"local":local,"external":external}
assert all(r["exit_code"] == 0 and r.get("excerpts") for r in external), [(r["id"],r["exit_code"]) for r in external]
target = BASE/"completion-evidence.json"
with target.open("x", encoding="utf-8") as handle:
    json.dump(supplement,handle,ensure_ascii=False,indent=2)
    handle.write("\n")

addition = '''
completion = json.loads((ROOT/BASE/"completion-evidence.json").read_text())
extra_artifacts = [
    {"id":"a-completion-code","kind":"code","path":(BASE/"complete_report.py").as_posix(),"sha256":digest(BASE/"complete_report.py"),"description":"保留既有檔案，以新收據補齊發布歷史、LoRA範例及學習者來源的最終彙整程式。"},
    {"id":"a-completion","kind":"execution","path":(BASE/"completion-evidence.json").as_posix(),"sha256":digest(BASE/"completion-evidence.json"),"description":"歷史實驗plan/Flash原始結果及LoRA身份／領域、LR日記和梯度提問的本回合原始收據，另列既有證據SHA。","command":completion["command"],"result":"三個原始HTTPS來源皆curl exit 0；讀取30組實驗及9項補充實驗的plan，確認Flash與reward/capstone保留；既有檔案未改寫。","environment":completion["environment"]},
]
report["artifacts"].extend(extra_artifacts)
report["sources"].append({"id":"s-completion","kind":"execution","title":"本回合補充原始來源與歷史實驗定位","verified":True,"artifact_id":"a-completion"})
for record in completion["external"]:
    report["sources"].append({"id":"s-completion-"+record["id"],"kind":"official_source","title":record["id"]+"：作者原始記錄","url":record["url"],"version":record["version"],"verified":True,"checked_original":True,"accessed_on":record["accessed_on"],"authority_reason":"原始專案作者的文件或提問／重現者本人的公開原文；只證明相應範例、提問與其自述觀察，不證明普遍效益或盛行率。","inspection_note":"直接curl讀取原文並檢查completion-evidence.json對應id的定位摘錄，未保存第三方全文。"})
for item in report["claims"]:
    if item["statement"].startswith("歷史實驗紀錄保留"):
        text = "plan.json實際列30組sequence與9項supporting_experiments，含flash_probe、reward/PPO與capstone系列；flash_probe原始結果記錄CUDA裝置與Flash探查。舊整合／學生結果與v2各自保留。"
        item["evidence"] = [{"source_id":"s-completion","locator":"completion-evidence.local：plan.json supporting_experiments及results/flash_probe.json","supports":text}]
        item["artifact_ids"] = ["a-completion","a-local"]
        item["verification"]["observed"] = text
        item["verification"]["details"] = "執行讀取原始plan及Flash結果並保存SHA／選定欄位，未重跑實驗。"
    if item["statement"].startswith("MiniMind提供LoRA"):
        item["artifact_ids"].append("a-completion")
        item["evidence"].append({"source_id":"s-completion-minimind-lora-examples","locator":"README lora_identity、lora_medical及私有領域範例","supports":"固定README直接提供身份與醫療領域的LoRA範例；不是由LoRA通用能力推測。"})
    if item["statement"].startswith("公開提問確實包含"):
        item["artifact_ids"].append("a-completion")
        for key, locator, supports in [("learner-lr-diary","CHANGELOG learning-rate/Muon段落","LR搜尋與超參數混雜的設計動機有重現者自述原文。"),("learner-gradient-question","Discussion856 backward/optimizer段落","梯度流程的設計動機有公開提問原文。")]:
            item["evidence"].append({"source_id":"s-completion-"+key,"locator":locator,"supports":supports})
        item["scope"] = "只支持定性的公開問題和自述觀察存在；不推定發文者在校身分、困難盛行率，或單一配置造成失敗的普遍因果。"
for item in report["artifacts"]:
    if item["id"] == "a-report-code":
        item["description"] = "最初彙整判斷的程式；本報告由complete_report.py在記憶體補齊來源定位後產生，此檔未改寫。"
for filename, expected in completion["preserved_inputs"].items():
    assert digest(BASE/filename) == expected, filename
'''
original = (BASE/"write_report.py").read_text()
marker = 'assert digest("docs/course-review-20261019/freeze-04/sources/curriculum.md") == report["source_sha256"]'
assert original.count(marker) == 1
prepared = original.replace(marker, addition+"\n"+marker)
exec(compile(prepared,str(BASE/"write_report.py"),"exec"),{"__name__":"__main__"})
