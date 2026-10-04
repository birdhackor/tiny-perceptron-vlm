"""Read bounded existing safetensors headers; no model load or weight download."""
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
original=ROOT/'outputs/natural-v4/modal-runs/prepare-37212856741/natural-natural-v4-prepare-37212856741-1/review/result.json'
j=json.loads(original.read_bytes())
cache=ROOT/'outputs/natural-v4/student-base-cache/hf'
results=[];files=0;total=0
for name,snapshot in j['snapshots'].items():
    path=cache/('models--'+snapshot['model'].replace('/','--'))/'snapshots'/snapshot['revision']
    assert {p.name for p in path.iterdir()}=={p['name'] for p in snapshot['files']}
    for item in snapshot['files']:
        assert (path/item['name']).stat().st_size==item['bytes']
        files+=1;total+=item['bytes']
    with (path/'model.safetensors').open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        assert length<1024*1024
        raw=f.read(length)
    header=json.loads(raw)
    count=sum(math.prod(t['shape']) for k,t in header.items() if k!='__metadata__')
    assert count==({'core':2127532032,'asr':808878080}[name])
    results.append({'model':snapshot['model'],'revision':snapshot['revision'],'header_bytes':length,
                    'header_sha256':hashlib.sha256(raw).hexdigest(),'unique_stored_tensor_count':len(header)-int('__metadata__' in header),
                    'sum_shape_elements':count,'model_file_bytes':(path/'model.safetensors').stat().st_size})
assert files==23 and total==5889111977
out={'original_record':str(original.relative_to(ROOT)), 'original_record_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
     'snapshot_file_count':files,'snapshot_bytes':total,'GiB':round(total/1024**3,4),'models':results,
     'scope':'Original preparation inventory cross-checked against existing selected cache file names/sizes and bounded inert headers. No new download, all-weight SHA recomputation, model load or benchmark replication. Snapshot size excludes environment, other caches and filesystem overhead.'}
print(json.dumps(out,indent=2))
