import json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
source,output,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path.insert(0,str(source))
from tiny_perceptron.omni_products import training_runtime as trainer
from tiny_perceptron.omni_products.inference import load_trained
config={"seed":13,"updates":4,"training_seconds":20,"batch_size":2,"learning_rate":0.001,"validate_every":1,"representation":"additive","variants":1,"width":16,"review_scope":"tiny train/dev continuation, not formal"}
value={"mode":mode,"source":str(source),"loaded_source_binding":trainer.LOADED_SOURCE_BINDING}
try:
 if mode=="partial":
  original=trainer.run_updates
  def partial(model,make_batch,*args,**kwargs):
   counter=[0]
   def batch(generator):
    counter[0]+=1
    return make_batch(generator)
   with patch.object(trainer,"time",SimpleNamespace(monotonic=lambda:100.0 if counter[0]>=2 else 0.0)):
    return original(model,batch,*args,**kwargs)
  with patch.object(trainer,"run_updates",side_effect=partial):
   result=trainer.train_av(config,output,"cpu")
 else:
  resume=(Path(sys.argv[4]),sys.argv[5]) if mode=="resumed" else None
  result=trainer.train_av(config,output,"cpu",resume=resume)
 value.update(accepted=True,complete=result["complete"],updates=result["updates"],source_unchanged=result["source_unchanged_during_training"],implementation_sha256=result["implementation_sha256"],latest_parameter_sha256=result["latest_parameter_sha256"])
 if mode!="partial":
  value["load_trained_accepted"]=load_trained(output,"av") is not None
except Exception as error:
 value.update(accepted=False,error_type=type(error).__name__,reason=str(error))
(output.parent/(mode+"-result.json")).write_text(json.dumps(value,indent=2)+"\n")
print("REVIEW_RESULT="+json.dumps(value))
