import math,torch,json
from tiny_perceptron.alignment import reliability_bins
c=torch.tensor([.55,.65,.85,.95,.25]);a=torch.tensor([1,0,1,0,1]);bins=reliability_bins(c,a,bins=2);ece=sum(b['count']/len(c)*abs(b['accuracy']-b['confidence']) for b in bins)
assert [b['count'] for b in bins]==[1,4] and abs(ece-.35)<1e-6
print(json.dumps({'bins':bins,'ece':ece,'rounded':round(ece,2),'nll_examples':[-math.log(.8),-math.log(.2)]},ensure_ascii=False,indent=2))
