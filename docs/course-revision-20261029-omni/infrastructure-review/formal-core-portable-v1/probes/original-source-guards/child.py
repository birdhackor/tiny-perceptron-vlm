import json,sys
from dataclasses import asdict
from pathlib import Path
source,output=Path(sys.argv[1]),Path(sys.argv[2]);sys.path.insert(0,str(source))
import torch
from tiny_perceptron.omni_products import training_runtime as t
from tiny_perceptron.omni_products.sound import SoundCodec,CodecConfig
from tiny_perceptron.omni_products.inference import load_trained
av=source/"tiny_perceptron/omni_products/av.py";original=av.read_bytes();rows=[]
config={"seed":7,"updates":1,"training_seconds":20,"batch_size":4,"learning_rate":0.001,"validate_every":1,"model":asdict(CodecConfig(hop=80,width=16,codes=32))}
for phase in ["stable_fresh","mutated_after_import","mutated_during_update"]:
 av.write_bytes(original);t.deterministic(7);model=SoundCodec(CodecConfig(hop=80,width=16,codes=32))
 if phase=="mutated_after_import":av.write_bytes(original+b"\n# review-only mutation after import\n")
 def batch(g):
  if phase=="mutated_during_update":av.write_bytes(original+b"\n# review-only mutation during update\n")
  return torch.randn(4,80,generator=g)*0.2
 try:
  r=t.run_updates(model,batch,lambda m,x:m(x),lambda _:0.0,config,output/phase)
  rows.append({"phase":phase,"accepted":True,"complete":r["complete"],"source_unchanged":r["source_unchanged_during_training"],"loader_accepted":load_trained(output/phase,"codec") is not None})
 except Exception as e:
  rows.append({"phase":phase,"accepted":False,"error_type":type(e).__name__,"reason":str(e),"result_file_exists":(output/phase/"training-result.json").exists()})
av.write_bytes(original)
print("REVIEW_RESULT="+json.dumps(rows))
