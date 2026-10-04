from pathlib import Path
import contextlib
import io
import json
import sys
import torch

torch.set_num_threads(1)
torch.set_default_device("cpu")
artifact = Path("docs/reader-reviews/artifacts/natural-v4-cleanup/T.6")
variants = []
raw_audio = (artifact / "prerequisite-12.8/fence-1.py").read_text()
variants.append(("12.8-width12", raw_audio.replace("width=8", "width=12")))
variants.append(("12.8-bands32-default-display", raw_audio.replace("AudioEncoder(bands=16", "AudioEncoder(bands=32")))
variants.append(("12.8-bands32-matched-display", raw_audio.replace("AudioEncoder(bands=16", "AudioEncoder(bands=32").replace("log_mel(wave)", "log_mel(wave, bands=32)")))
raw_vision = (artifact / "prerequisite-10.5/fence-1.py").read_text()
variants.append(("10.5-swapped-labels", raw_vision.replace("torch.tensor([0, 1])", "torch.tensor([1, 0])")))
results = []
for label, code in variants:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, label, "exec"), {"__name__":"__main__"})
    results.append({"variant":label,"source_code":code,"stdout":output.getvalue(),"result":"completed without parameter update"})
print(json.dumps({"scope":"Bounded offline CPU prerequisite exercise variants only", "environment":{"python":sys.version,"torch":str(torch.__version__),"cuda_build":str(torch.version.cuda),"cuda_available":str(torch.cuda.is_available()),"device":"cpu","threads":"1"},"results":results},ensure_ascii=False,indent=2))
