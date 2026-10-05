import hashlib, json, math, platform, sys
from pathlib import Path
import torch

torch.set_num_threads(1)
torch.set_default_device("cpu")
code_path=Path(__file__).with_name("original-fence-1.py")
namespace={"__name__":"independent_bounded_check"}
exec(compile(code_path.read_bytes(),str(code_path),"exec"),namespace)
simulate=namespace["simulate"]
x=namespace["evaluation"]
fixed=namespace["fixed_max"]
dynamic=namespace["dynamic_max"]
assert x.shape==(3,) and x.dtype==torch.float32
assert fixed.shape==dynamic.shape==torch.Size([])
assert fixed.item()==1 and dynamic.item()==3
expected={"fixed":[-1/7,3/7,1],"dynamic":[0,3/7,3]}
result={"environment":{"python":sys.version,"torch":str(torch.__version__),"torch_git_version":torch.version.git_version,"device":"cpu","cuda_build":str(torch.version.cuda),"default_dtype":str(torch.get_default_dtype())},"evaluation_count":x.numel(),"calibration_count":namespace["calibration"].numel(),"original_code_sha256":hashlib.sha256(code_path.read_bytes()).hexdigest(),"checks":{}}
for name,maximum in (("fixed",fixed),("dynamic",dynamic)):
 scale=maximum/7
 raw_codes=(x/scale).round()
 codes=raw_codes.clamp(-7,7)
 restored=simulate(x,maximum)
 torch.testing.assert_close(restored,torch.tensor(expected[name]),rtol=0,atol=1e-7)
 assert int((x.abs()>maximum).sum())==({"fixed":1,"dynamic":0}[name])
 assert restored.dtype==x.dtype and restored.device.type=="cpu"
 assert not restored.requires_grad
 result["checks"][name]={"maximum":maximum.item(),"scale":scale.item(),"raw_codes":raw_codes.tolist(),"clamped_codes":codes.tolist(),"restored":restored.tolist(),"overflow_count":int((x.abs()>maximum).sum()),"restored_shape":list(restored.shape),"restored_dtype":str(restored.dtype),"absolute_error":(restored-x).abs().tolist()}
calibration_plus_three=torch.cat([namespace["calibration"],torch.tensor([3.0])])
new_fixed=calibration_plus_three.abs().max()
exercise=simulate(x,new_fixed)
assert new_fixed.item()==3 and torch.equal(exercise,simulate(x,dynamic))
result["checks"]["exercise"]={"calibration_count":5,"fixed_max":new_fixed.item(),"output":exercise.tolist(),"equals_dynamic":True}
# Exact binary half-grid probes verify PyTorch's tie-to-even rule separately from float32 thirds.
probe=torch.tensor([-8.,-7.,-.5,.5,1.5,2.5,7.,8.])
p=simulate(probe,torch.tensor(7.))
assert torch.equal(p,torch.tensor([-7.,-7.,0.,0.,2.,2.,7.,7.]))
result["checks"]["rounding_and_clamp"]={"input":probe.tolist(),"output":p.tolist(),"grid_step":1.0,"levels":15}
# A nearest-rank empirical percentile is used solely as a bounded, exact coverage example.
a=torch.tensor([.5]*99+[100.])
rank=math.ceil(.99*a.numel())
threshold=a.abs().sort().values[rank-1]
assert threshold.item()==.5
assert int((a.abs()<=threshold).sum())==99
assert int((a.abs()>threshold).sum())==1
assert threshold/7<a.abs().max()/7
result["checks"]["percentile_illustration"]={"definition":"nearest-rank empirical 99th percentile of 100 absolute values, not an ONNX histogram API equivalence test","count":100,"retained_count":99,"clipped_count":1,"coverage":.99,"threshold":threshold.item(),"percentile_scale":(threshold/7).item(),"minmax_scale":(a.abs().max()/7).item()}
# Input amplitudes alter the observed maximum even though no weight is trained.
assert torch.equal((10*x).abs().max(),10*dynamic)
result["checks"]["amplitude_change"]={"original_max":dynamic.item(),"tenfold_input_max":(10*x).abs().max().item()}
# Counterexample to a categorical claim that every unrepresentative calibration distribution clips frequently.
broad_calibration=torch.tensor([-5.,2.,8.,10.])
broad_max=broad_calibration.abs().max()
broad_output=simulate(x,broad_max)
assert broad_max.item()==10.0
assert int((x.abs()>broad_max).sum())==0
assert (broad_max/7).item()>(dynamic/7).item()
result["checks"]["unrepresentative_broad_range"]={"calibration":broad_calibration.tolist(),"evaluation":x.tolist(),"fixed_max":broad_max.item(),"scale":(broad_max/7).item(),"clipped_count":int((x.abs()>broad_max).sum()),"evaluation_count":x.numel(),"restored":broad_output.tolist(),"interpretation":"Calibration values are much broader than actual evaluated activations, yet there is zero clipping; unrepresentative calibration can cause an unnecessarily coarse grid instead of frequent clipping."}
result["scope"]="Original scalar per-tensor float32 simulation; no model parameters, integer storage, backward, optimizer, training, GPU, model download, or model score evaluation."
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
