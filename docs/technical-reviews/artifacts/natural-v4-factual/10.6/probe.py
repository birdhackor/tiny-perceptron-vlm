from pathlib import Path
import contextlib,hashlib,io,json,sys,torch
from torch import nn
out=Path(__file__).parent
predictions={
 'main_shapes': [[1,16,8],[1,16,12],[12,8]],
 'exercise_shapes': [[1,16,10],[10,8]],
 'same_seed_rerun':'same tensors, weights, bias, outputs in this same runtime/device/order',
 'shared_affine':'every [batch,position] uses same W,b; output equals vision @ W.T + b',
 'wrong_widths':'10 or 16 cannot be fed to a fixed Linear(in_features=12), nor added to [1,16,12]',
 'expansion_example':'[[1,0],[0,1],[1,1]] @ [1,2] = [1,2,3]',
 'identical_feature_collision':'the same input, W, bias produces equal outputs for width 12 and 100',
 'prerequisite_W5':'matrix [[40,1],[80,6]], biased [[45,1],[85,6]], exercise [[48,1],[88,6]]',
 'prerequisite_10_3':'input [1,16,48], output [1,16,8], W [8,48]; exercise output [1,16,16], W [16,48]'
}
(out/'predictions.json').write_text(json.dumps(predictions,indent=2)+'\n')
results={}

def forward():
 torch.manual_seed(0)
 vision=torch.randn(1,16,8)
 projector=nn.Linear(8,12)
 return vision,projector,projector(vision)
vision,projector,language_features=forward()
print('視覺',tuple(vision.shape)); print('文字介面',tuple(language_features.shape)); print('接頭矩陣',tuple(projector.weight.shape))
results['main_shapes']=[list(vision.shape),list(language_features.shape),list(projector.weight.shape)]
assert results['main_shapes']==predictions['main_shapes']
manual=vision @ projector.weight.T + projector.bias
torch.testing.assert_close(language_features,manual,rtol=1e-6,atol=1e-6)
# A single common vector at all positions must get a single common output vector.
repeated=vision[:,0:1,:].expand(1,16,8)
assert torch.equal(projector(repeated),projector(repeated)[:,0:1,:].expand(1,16,12))
results['shared_affine']={'max_abs_error':float((language_features-manual).abs().max().detach()),'positions':16,'input_width':8,'output_width':12}
vision2,projector2,output2=forward()
assert torch.equal(vision,vision2) and torch.equal(projector.weight,projector2.weight) and torch.equal(projector.bias,projector2.bias) and torch.equal(language_features,output2)
results['same_seed_rerun_equal']=True
exercise=nn.Linear(8,10)
exercise_output=exercise(vision)
results['exercise_shapes']=[list(exercise_output.shape),list(exercise.weight.shape)]
assert results['exercise_shapes']==predictions['exercise_shapes']
errors={}
for width in (10,16):
 x=nn.Linear(8,width)(vision)
 for op in ('downstream_linear','addition'):
  try:
   if op=='downstream_linear': nn.Linear(12,3)(x)
   else: x+torch.zeros(1,16,12)
  except RuntimeError as e: errors[f'{width}/{op}']=str(e)
  else: raise AssertionError(f'{width}/{op} unexpectedly succeeded')
results['expected_shape_errors']=errors
W=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
x=torch.tensor([1.,2.]); y=W@x
assert y.tolist()==[1.,2.,3.]
results['expansion_example']={'input':x.tolist(),'weight':W.tolist(),'output':y.tolist(),'weight_rank':int(torch.linalg.matrix_rank(W))}
collapsed=torch.tensor([[0.5]*8,[0.5]*8])
collisions={}
for width in (12,100):
 layer=nn.Linear(8,width)
 z=layer(collapsed)
 assert torch.equal(z[0],z[1])
 collisions[str(width)]={'identical_input':True,'equal_output':True,'max_abs_difference':float((z[0]-z[1]).abs().max().detach())}
results['feature_collision']=collisions
recipes=torch.tensor([[2.,1.,0.],[1.,2.,1.]])
weights=torch.tensor([[10.,0.],[20.,1.],[30.,4.]])
recipe_output=recipes@weights
layer=nn.Linear(3,2)
with torch.no_grad(): layer.weight.copy_(weights.T); layer.bias.copy_(torch.tensor([5.,0.]))
biased=layer(recipes)
with torch.no_grad(): layer.bias.copy_(torch.tensor([8.,0.]))
exercise_biased=layer(recipes)
assert recipe_output.tolist()==[[40.,1.],[80.,6.]]
assert biased.tolist()==[[45.,1.],[85.,6.]]
assert exercise_biased.tolist()==[[48.,1.],[88.,6.]]
results['W5']={'matrix':recipe_output.tolist(),'biased':biased.tolist(),'exercise':exercise_biased.tolist()}
from tiny_perceptron.multimodal import scene,patchify
torch.manual_seed(0)
patches=patchify(scene()[None],4); embedding=nn.Linear(48,8); features=embedding(patches)
assert list(patches.shape)==[1,16,48] and list(features.shape)==[1,16,8] and list(embedding.weight.shape)==[8,48]
embedding16=nn.Linear(48,16); features16=embedding16(patches)
assert list(features16.shape)==[1,16,16] and list(embedding16.weight.shape)==[16,48]
results['10.3']={'input_shape':list(patches.shape),'output_shape':list(features.shape),'weight_shape':list(embedding.weight.shape),'exercise_output_shape':list(features16.shape),'exercise_weight_shape':list(embedding16.weight.shape)}
results['environment']={'python':sys.version,'torch':torch.__version__,'device':'cpu','cuda_available':torch.cuda.is_available()}
results['limits']='Small deterministic CPU arithmetic/API checks only; no paired-image training, GPU training, model download, quality benchmark or speed measurement.'
(out/'probe-results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
