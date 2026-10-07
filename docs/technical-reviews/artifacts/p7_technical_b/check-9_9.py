import torch,json
scores=torch.tensor([[2.0,1.0,0.0]]);correct_index=1;out=[]
for scale in [1.0,10.0]:
    probability=(scores*scale).softmax(-1);confidence,prediction=probability.max(-1)
    print('倍率',scale,'信心',round(confidence.item(),4),'選擇',prediction.item(),'正確',prediction.item()==correct_index)
    assert prediction.item()==0 and not prediction.item()==correct_index
    out.append({'scale':scale,'probabilities':probability.tolist(),'confidence':confidence.item()})
probability=(scores*.5).softmax(-1);confidence,prediction=probability.max(-1)
print('variation .5',round(confidence.item(),4),prediction.item(),prediction.item()==correct_index)
assert prediction.item()==0 and confidence.item()<out[0]['confidence']
print(json.dumps({'original':out,'half':probability.tolist(),'torch':torch.__version__,'device':'cpu'}))
