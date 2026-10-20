"""補查獨立來源、增強檔案、教學入口與引用連結的實際範圍。"""

import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def fetch(url):
    argv=["curl","--location","--fail","--silent","--show-error","--max-time","90",url]
    p=subprocess.run(argv,capture_output=True,check=True)
    return p.stdout,{"command":argv,"bytes":len(p.stdout),"sha256":hashlib.sha256(p.stdout).hexdigest()}


def main():
    manifest=json.loads((ROOT/"docs/selftrained/v2-manifest.json").read_bytes())
    p=manifest["package"]
    raw,receipt=fetch(f"https://media.githubusercontent.com/media/{p['repo_id']}/{p['revision']}/{p['path']}")
    result={"command":"PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-training-assets/details.py",
            "environment":{"python":sys.version,"device":"cpu","network":"curl read-only HTTPS"},"selftrained_archive":receipt,
            "voice":{},"ocr":{},"vision":{}}
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as tar:
        for item in manifest["records"]:
            name=item["path"]
            data=[json.loads(x) for x in tar.extractfile(name).read().splitlines() if x]
            if name.startswith("voice-"):
                audio={r["audio"] for r in data}
                original={a for a in audio if "/voice-augmentation-v2/" not in a}
                derived=audio-original
                result["voice"][name]={"rows":len(data),"unique_wav":len(audio),"original_wav":len(original),"derived_wav":len(derived),
                                       "groups":len({r["group_id"] for r in data}),"intents":dict(Counter(r["supervision"]["intent"] for r in data)),
                                       "speaker_fields_present":any("speaker" in k for r in data for k in r)}
            if name.startswith("ocr-"):
                answers=[r["supervision"]["ocr_text"] for r in data]
                result["ocr"][name]={"characters":sorted(set("".join(answers))),"character_count":len(set("".join(answers))),
                                     "answer_lengths":dict(Counter(map(len,answers))),"font_families":dict(Counter(r["supervision"]["font_family"] for r in data)),
                                     "all_have_supplied_roi":all(len(r["roi"])==4 for r in data)}
            if name.startswith("vision-"):
                result["vision"][name]={"rows":len(data),"supervision_keys":sorted({k for r in data for k in r["supervision"]}),
                                        "tasks":dict(Counter(r["task"] for r in data)),"first_supervision":data[0]["supervision"]}
        d=json.loads(tar.extractfile("provenance/v2/voice-augmentation-v2.json").read())
        result["voice_augmentation_provenance"]={"keys":list(d),"summary":{k:v for k,v in d.items() if not isinstance(v,list) and not isinstance(v,dict)}}
    result["links"]=[]
    page=(ROOT/"assets/training/README.md").read_text()
    for label,url in re.findall(r"\[([^\]]+)\]\(([^)]+)\)",page):
        if url.startswith("https:"):
            continue
        path,_,anchor=url.partition("#")
        target=(ROOT/"assets/training"/path).resolve()
        result["links"].append({"label":label,"path":str(target.relative_to(ROOT)),"exists":target.is_file(),"anchor":anchor,
                                "anchor_appears_as_heading_prefix":not anchor or any(line.startswith("#") and anchor in line for line in target.read_text().splitlines())})
    result["notebook_data_dependencies"]={}
    notebooks=list((ROOT/"notebooks").glob("*.ipynb"))
    result["notebook_count"]=len(notebooks)
    for path in notebooks:
        data=json.loads(path.read_bytes())
        code="\n".join("".join(cell["source"]) for cell in data["cells"] if cell["cell_type"]=="code")
        matches=[line for line in code.splitlines() if any(term in line for term in ["data/training","fetch_training_assets","assets/training"])]
        if matches:
            result["notebook_data_dependencies"][str(path.relative_to(ROOT))]=matches
    originals=[
        ("fsdd-license","https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/26eb9aaf76e81b692f806f9140c2d2777410d7a1/README.md",r"https://creativecommons.org/licenses/by-sa/4.0/"),
        ("minds-license","https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md",r"- cc-by-4.0"),
        ("noto-license","https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/LICENSE",r"SIL OPEN FONT LICENSE Version 1.1.*"),
        ("lfs-config-current","https://raw.githubusercontent.com/git-lfs/git-lfs/v3.8.0/docs/man/git-lfs-config.adoc",r"Sets whether or not.*"),
        ("group-split","https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/sklearn/model_selection/_split.py",r".*The same group will not appear in two different folds.*"),
    ]
    result["original_sources"]=[]
    for name,url,pattern in originals:
        raw,receipt=fetch(url)
        txt=raw.decode()
        excerpts=[]
        for number,line in enumerate(txt.splitlines(),1):
            if re.search(pattern,line):
                excerpts.append({"line":number,"excerpt":" ".join(line.split()[:20])})
                break
        if name=="lfs-config-current":
            lines=txt.splitlines();i=next(i for i,line in enumerate(lines) if "`GIT_LFS_SKIP_SMUDGE`" in line)
            excerpts=[{"line":i+1,"excerpt":" ".join(" ".join(lines[i:i+8]).split()[:24])}]
        result["original_sources"].append({"id":name,"url":url,"receipt":receipt,"excerpts":excerpts,"accessed_on":"2026-10-20"})
    path=OUT/"details-output.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"voice":result["voice"],"ocr":result["ocr"],"sources":result["original_sources"],"notebook_count":len(notebooks),"notebook_data_dependencies":result["notebook_data_dependencies"]},ensure_ascii=False))


if __name__=="__main__":
    main()
