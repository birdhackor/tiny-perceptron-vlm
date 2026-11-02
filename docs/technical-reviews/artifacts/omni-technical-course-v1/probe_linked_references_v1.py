"""Run only the explicitly linked CPU mechanism references backing course R.4."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import torch

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path("/workspace/work/reference/technical-course-20261029")
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    freeze = json.loads(FREEZE.read_text())
    results = {"created_at":datetime.now(timezone.utc).isoformat(),
               "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu"},
               "scope":"Exact embedded CPU mechanism code only. 19.21 backward with no update; 19.22 one illustrative optimizer step; 19.23 random forward/calibration fixture. No formal training, mature models, GPU, downloaded weights or capability acceptance.",
               "lessons":{}}
    os.chdir(CACHE)
    for lesson_id in ["19.21","19.22","19.23"]:
        page = next(p for p in freeze["pages"] if p["page_id"] == lesson_id)
        snapshot = ROOT / page["snapshot"]
        assert sha(snapshot) == page["source_sha256"]
        blocks = re.findall(r"```python\n(.*?)```",snapshot.read_text(),re.S)
        namespace = {}
        stream = io.StringIO()
        with redirect_stdout(stream):
            for i,code in enumerate(blocks):
                exec(compile(code,f"{snapshot}:block-{i+1}","exec"),namespace)
        results["lessons"][lesson_id] = {"snapshot":page["snapshot"],"source_sha256":page["source_sha256"],
            "code_block_sha256s":[hashlib.sha256(code.encode()).hexdigest() for code in blocks],
            "stdout":stream.getvalue(),"completed":True}
        if lesson_id == "19.21":
            results["lessons"][lesson_id]["record"] = namespace["record"]
        if lesson_id == "19.22":
            wav = CACHE / "demo-sound.wav"
            results["lessons"][lesson_id]["generated_wav_outside_git"] = {"path":str(wav),"sha256":sha(wav),"bytes":wav.stat().st_size} if wav.is_file() else None
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
