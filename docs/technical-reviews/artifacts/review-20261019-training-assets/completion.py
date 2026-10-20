"""遞迴核對 Notebook 依賴、實驗紀錄，並實跑小型自製快照重建。"""

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
os.environ["PYTHONDONTWRITEBYTECODE"]="1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result={"command":"PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-training-assets/completion.py",
            "environment":{"python":sys.version,"device":"cpu"},"notebooks":{},"experiment_records":[],"entry_points":[]}
    notebooks=list((ROOT/"notebooks").rglob("*.ipynb"))
    result["notebooks"]={"recursive_count":len(notebooks),"matches":{},"correction":"details-output.json 的 notebook_count 僅列頂層檔案，未涵蓋章目錄；本收據遞迴枚舉完整範圍。"}
    for path in notebooks:
        d=json.loads(path.read_bytes())
        lines=[line for c in d["cells"] if c["cell_type"]=="code" for line in "".join(c["source"]).splitlines()]
        matches=[line for line in lines if any(term in line for term in ["data/training","fetch_training_assets","assets/training"])]
        if matches:
            result["notebooks"]["matches"][str(path.relative_to(ROOT))]=matches
    plan=json.loads((ROOT/"docs/course-experiments/plan.json").read_bytes())
    def scan(value,path="results"):
        found=[]
        if isinstance(value,dict):
            if any(k in value for k in ["train","validation","test"]) and any(k in path for k in ["data","split"]):
                found.append({"path":path,"split_keys":[k for k in value if k in ["train","validation","test"]],
                              "value":{k:v for k,v in value.items() if k in ["train","validation","test"] and not isinstance(v,list)}})
            for k,v in value.items():
                found+=scan(v,path+"."+k)
        elif isinstance(value,list) and len(value)<50:
            for i,v in enumerate(value):
                found+=scan(v,path+f"[{i}]")
        return found
    for e in plan["sequence"]:
        path=ROOT/e["evidence"]
        d=json.loads(path.read_bytes())
        split_artifacts=[a for a in d["artifacts"] if any(term in a["path"] for term in ["data","manifest","split","resampl"])]
        result["experiment_records"].append({"id":e["id"],"path":e["evidence"],"sha256":digest(path),"status":d.get("evidence_status"),
                                              "revision":d["revision"],"dependencies":e["dependencies"],
                                              "split_metadata":scan(d["results"]),"data_artifact_count":len(split_artifacts),"sample_data_artifacts":split_artifacts[:8]})
    for argv in [[sys.executable,"scripts/build_training_assets.py","--help"],
                 [sys.executable,"-m","scripts.course_experiments.run","--help"],
                 [sys.executable,"scripts/selftrained/train_local_stage.py","--help"]]:
        p=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
        result["entry_points"].append({"command":argv,"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr})
    # 用自製短字串填滿所有必要來源路徑，實際執行文件指定的 builder。
    with tempfile.TemporaryDirectory(dir=OUT,prefix="rebuild-synthetic-") as temp:
        work=Path(temp);source=work/"source"
        dirs=["text-initial","vision-initial","reasoning-initial","fsdd-initial"]+["behavior-initial/"+n for n in ["ultrachat-sft","ultrafeedback-dpo","pku-safe-rlhf"]]
        for name in dirs:
            (source/name).mkdir(parents=True)
            (source/name/"README.md").write_text("自製測試材料\n")
        (source/"text-initial/asset-manifest.json").write_text(json.dumps({"assets":[{"license":"CDLA-Sharing-1.0"},{"license":"MIT"}]}))
        (source/"text-initial/tinystories-example.jsonl").write_text('{"text":"story"}\n')
        (source/"text-initial/chinese-example.jsonl").write_text('{"text":"詩"}\n')
        (source/"text-initial/tang300-source.json").write_text("[]")
        for directory,metadata in [("vision-initial","manifest.json"),("reasoning-initial","asset-manifest.json"),("fsdd-initial","asset.json")]+[("behavior-initial/"+n,"asset.json") for n in ["ultrachat-sft","ultrafeedback-dpo","pku-safe-rlhf"]]:
            (source/directory/metadata).write_text(json.dumps({"license":"MIT"}))
        commands=[]
        for dest in [work/"one",work/"two"]:
            argv=[sys.executable,"scripts/build_training_assets.py","--source",str(source),"--output",str(dest)]
            p=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True)
            commands.append({"command":[str(x).replace(str(work),"<本 artifact 目錄內已清除的自製暫存資料>") for x in argv],"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr})
        result["synthetic_builder"]={"executions":commands,"manifest_bytes_identical":(work/"one/manifest.json").read_bytes()==(work/"two/manifest.json").read_bytes(),
                                     "archives_identical":{p.name:digest(p)==digest(work/"two"/p.name) for p in (work/"one").glob("*.tar.gz")},
                                     "scope":"僅驗證命令與固定編碼，不用自製來源證明正式資料筆數；正式八包另以 audit-output.json 的逐檔內容重建比對。"}
    result["extra_source_hashes"]={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/"scripts/course_experiments/behavior.py",ROOT/"scripts/course_experiments/run.py",ROOT/"scripts/prepare_data.py",ROOT/"scripts/selftrained/prepare_voice.py",ROOT/"scripts/selftrained/prepare_vision_ocr.py",ROOT/"scripts/selftrained/augment_voice_v2.py",ROOT/"scripts/selftrained/augment_ocr_v2.py",ROOT/"scripts/selftrained/train_local_stage.py"]}
    path=OUT/"completion-output.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"path":str(path.relative_to(ROOT)),"sha256":digest(path),"notebooks":result["notebooks"],"experiments":len(result["experiment_records"]),"builder":result["synthetic_builder"]},ensure_ascii=False))


if __name__=="__main__":
    main()
