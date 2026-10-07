import torch,json
from tiny_perceptron.alignment import reliability_bins
confidence=torch.tensor([.55,.65,.85,.95]);correct=torch.tensor([1,0,1,0]);bins=reliability_bins(confidence,correct,bins=2)
for row in bins:
    print('區間',row['left'],row['right'],'筆數',row['count'],'信心',round(row['confidence'],2),'正確率',row['accuracy'])
ece=sum(row['count']/len(confidence)*abs(row['accuracy']-row['confidence']) for row in bins);print('平均校準差',round(ece,2))
assert len(bins)==1 and bins[0]['count']==4 and abs(bins[0]['confidence']-.75)<1e-6 and bins[0]['accuracy']==.5 and abs(ece-.25)<1e-6
boundary=reliability_bins(torch.tensor([0.,.5,1.]),torch.tensor([0,1,1]),bins=2)
print('boundary variation',json.dumps(boundary));assert [x['count'] for x in boundary]==[1,2]
print('raw',json.dumps({'bins':bins,'ece':ece,'torch':torch.__version__,'device':'cpu'}))
