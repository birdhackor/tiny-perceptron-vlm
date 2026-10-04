import hashlib
import json
import struct
from pathlib import Path
import soundfile as sf

root=Path(__file__).resolve().parents[5]
p=root/'outputs/natural-v4/factual-research/T.1/own-first-derived.wav'
values,rate=sf.read(p,dtype='float32')
fresh=root/'outputs/natural-v4/factual-research/T.1/own-second-derived.wav'
sf.write(fresh,values,rate,subtype='FLOAT')
a,b=p.read_bytes(),fresh.read_bytes()
diffs=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
offset=12; chunks=[]
while offset+8<=len(a):
    name=a[offset:offset+4].decode(); size=struct.unpack_from('<I',a,offset+4)[0]
    c={'name':name,'offset':offset,'bytes':size}
    if name=='PEAK':
        stampoff=offset+12
        c['version']=struct.unpack_from('<I',a,offset+8)[0]
        c['first_write_time']=struct.unpack_from('<I',a,stampoff)[0]
        c['second_write_time']=struct.unpack_from('<I',b,stampoff)[0]
    if name=='data': c['payload_sha256']=hashlib.sha256(a[offset+8:offset+8+size]).hexdigest()
    chunks.append(c);offset+=8+size+(size%2)
assert len(a)==len(b) and diffs and all(stampoff<=i<stampoff+4 for i in diffs)
print(json.dumps({'file_bytes':len(a),'differing_byte_offsets':diffs,'chunks':chunks,
 'first_sha256':hashlib.sha256(a).hexdigest(),'second_sha256':hashlib.sha256(b).hexdigest(),
 'result':'Own identical-sample WAV writes differ only within PEAK write-time field. No timestamp/byte edits; original historical full-file SHA not independently reconstructed.'},indent=2))
