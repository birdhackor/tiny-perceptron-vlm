before = {k:v.detach().clone() for k,v in encoder.state_dict().items()}
original_grad = encoder.projection.weight.grad.detach().clone()
encoder.zero_grad();classifier.zero_grad()
swapped_loss = F.cross_entropy(classifier(encoder(images).mean(1)),torch.tensor([1,0]))
swapped_loss.backward()
print('swapped targets gradient positive',encoder.projection.weight.grad.norm().item()>0)
print('swapped targets gradient changed',not torch.equal(original_grad,encoder.projection.weight.grad))
print('no optimizer encoder weights unchanged',all(torch.equal(v,before[k]) for k,v in encoder.state_dict().items()))
