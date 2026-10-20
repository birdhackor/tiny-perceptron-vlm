"""以 curl 讀取固定原始來源，只留授權與資料定義的短摘錄。"""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ARTIFACT = Path(__file__).resolve().parent
SOURCES = [
    ("tinystories", "https://huggingface.co/datasets/roneneldan/TinyStories/raw/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md", r"license:.*"),
    ("chinese-poetry", "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE", r"MIT License|Permission is hereby granted.*"),
    ("ultrachat", "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/raw/8049631c405ae6576f93f445c6b8166f76f5505a/README.md", r"license:.*"),
    ("ultrafeedback", "https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/raw/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85/README.md", r"license:.*"),
    ("pku", "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/raw/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md", r"license:.*"),
    ("fashion", "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE", r"MIT License|Permission is hereby granted.*"),
    ("gsm-license", "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE", r"MIT License|Permission is hereby granted.*"),
    ("gsm-authorship", "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/README.md", r"GSM8K consists of.*"),
    ("fsdd", "https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/26eb9aaf76e81b692f806f9140c2d2777410d7a1/README.md", r".*8kHz.*|.*CC BY-SA.*|.*Creative Commons.*"),
    ("minds14", "https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md", r"license:.*"),
    ("lfs-config", "https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.1/docs/man/git-lfs-config.adoc", r".*GIT_LFS_SKIP_SMUDGE.*|.*skip.*smudge.*"),
    ("lfs-install", "https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.1/docs/man/git-lfs-install.adoc", r".*--local.*|.*local repository.*"),
    ("powershell", "https://raw.githubusercontent.com/MicrosoftDocs/PowerShell-Docs/main/reference/7.5/Microsoft.PowerShell.Core/About/about_Environment_Variables.md", r".*\$Env:.*=.*"),
    ("cc-noncommercial", "https://creativecommons.org/licenses/by-nc/4.0/deed.en", r".*NonCommercial.*"),
]


def main():
    result = {"command":"PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-training-assets/upstream.py",
              "environment":{"python":sys.version,"transport":"curl","storage":"全文僅在記憶體；收據只留短摘錄"},"accessed_on":"2026-10-20","sources":[]}
    for name,url,pattern in SOURCES:
        argv=["curl","--location","--fail","--silent","--show-error","--max-time","45",url]
        proc=subprocess.run(argv,capture_output=True)
        raw=proc.stdout
        text=raw.decode("utf-8",errors="replace")
        hits=[]
        budget=24
        for index,line in enumerate(text.splitlines(),1):
            if re.search(pattern,line,re.IGNORECASE) and budget>0:
                words=line.split()
                excerpt=" ".join(words[:min(budget,16)])
                hits.append({"line":index,"excerpt":excerpt})
                budget-=len(excerpt.split())
        result["sources"].append({"id":name,"url":url,"command":argv,"exit_code":proc.returncode,
                                  "bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"total_lines":len(text.splitlines()),
                                  "excerpts":hits,"stderr":proc.stderr.decode()})
    path=ARTIFACT/"upstream-output.json"
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "sources":[{"id":s["id"],"exit_code":s["exit_code"],"excerpts":s["excerpts"]} for s in result["sources"]]},ensure_ascii=False))


if __name__ == "__main__":
    main()
