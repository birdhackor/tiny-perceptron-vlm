more={5:torch.randn(5,8)}
xx,yy=expand_modalities(ids,labels,embedding,more,{5})
print('five image features',tuple(xx.shape),yy.tolist(),'ignored count',int((yy==-100).sum()))
print('A byte plus offset',ord('A'),ord('A')+8)
