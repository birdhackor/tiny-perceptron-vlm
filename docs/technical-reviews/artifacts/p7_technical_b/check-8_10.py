import torch,json
human=torch.tensor([1,0,1,0]);recorded_judge=torch.tensor([1,1,1,0]);agree=human==recorded_judge
print('一致比例',agree.float().mean().item());print('分歧索引',(~agree).nonzero().flatten().tolist())
assert agree.float().mean().item()==.75 and (~agree).nonzero().flatten().tolist()==[1]
changed=recorded_judge.clone();changed[1]=0;assert torch.equal(changed,human)
all_wrong=1-human;assert (all_wrong==human).sum()==0
print(json.dumps({'agreement_count':int(agree.sum()),'denominator':4,'corrected_index1_agreement':float((changed==human).float().mean()),'inverse_agreement':float((all_wrong==human).float().mean())}))
