other_ids=torch.tensor([1,40,50])
print('other ids shape/equal',tuple(lm(other_ids[None])['logits'].shape),torch.equal(lm(other_ids[None])['logits'],multimodal(other_ids)['logits']))
try:
    multimodal(torch.tensor([1,5,4]))
except ValueError as e:
    print('missing image ValueError',str(e))
print('TinyLM module types',sorted({type(m).__name__ for m in lm.modules()}))
lm.train()
print('TinyLM repeated train-mode forward equal',torch.equal(lm(ids[None])['logits'],lm(ids[None])['logits']))
