import json,math,torch
import torch.nn.functional as F
from pathlib import Path
vocab=["雨","停","了","〈E〉"]
next_labels=torch.tensor([0,1,2,3]);later_labels=torch.tensor([1,2,3,-100])
next_scores=torch.zeros(4,4);later_scores=torch.zeros(4,4)
next_loss=F.cross_entropy(next_scores,next_labels,ignore_index=-100);later_loss=F.cross_entropy(later_scores,later_labels,ignore_index=-100)
weight=.5;total=next_loss+weight*later_loss
out={"next_count":int((next_labels!=-100).sum()),"later_count":int((later_labels!=-100).sum()),"next_loss":next_loss.item(),"later_loss":later_loss.item(),"total":total.item(),"weight1_total":(next_loss+later_loss).item(),"hand_ln4":math.log(4)}
assert out["next_count"]==4 and out["later_count"]==3
assert abs(out["total"]-1.5*math.log(4))<1e-6
print(json.dumps(out,ensure_ascii=False))
