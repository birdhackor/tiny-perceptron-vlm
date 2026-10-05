import torch,soundfile as sf,inspect,hashlib,json
from pathlib import Path
out=[]
for obj,name in [(torch.nn.LayerNorm,'torch-normalization-source'),(torch.stft,'torch-functional-source'),(sf.read,'soundfile-source')]:
 p=Path(inspect.getsourcefile(obj));raw=p.read_bytes();out.append({'authority_id':name,'runtime_source_path':str(p),'runtime_sha256':hashlib.sha256(raw).hexdigest(),'retrieved_official_source_byte_identical':raw==Path('outputs/natural-v4/factual-research/T.6/'+name+'.original').read_bytes()})
print(json.dumps(out,indent=2))
