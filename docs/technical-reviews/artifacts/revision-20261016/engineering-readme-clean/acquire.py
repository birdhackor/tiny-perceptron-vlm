"""Read only named public original sources; full downloads stay in ignored work/."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[5]
WORK=ROOT/'work/engineering-readme-clean/official'
REV='979cdfacc588ad0536f1c64fff96f264571cf054'
SOURCES=[
 ('hf-tree',f'https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/{REV}/selftrained/v2?recursive=true&expand=false'),
 ('uv-sync-source','https://raw.githubusercontent.com/astral-sh/uv/0.12.22/docs/concepts/projects/sync.md'),
 ('torch-tensor-source','https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/_tensor.py'),
 ('github-macos-job','https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/actions/jobs/111052346622'),
 ('historical-env','https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/1ac7fcbfafca30a7ec1d46d82947a16f82fb7bbe/scripts/check_env.py'),
]
SOURCES += [('hf-'+group+'-manifest', f'https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/{REV}/selftrained/v2/{group}/inference-manifest.json')
            for group in ['moe-pretrain','moe-sft','moe-joint','dense-joint']]
if __name__=='__main__':
    WORK.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,url in SOURCES:
        with urlopen(url,timeout=25) as response:
            raw=response.read(); resolved=response.url
        path=WORK/name;path.write_bytes(raw)
        records.append({'id':name,'url':url,'resolved_url':resolved,'checked_at':datetime.now(UTC).isoformat(),
                        'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                        'local':str(path.relative_to(ROOT)),
                        'command':'PYTHONPATH=. .venv/bin/python '+str(Path(__file__).relative_to(ROOT))})
    (WORK/'download-receipts.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'downloaded':len(records),'credential_use':False,'remote_mutations':False}))
