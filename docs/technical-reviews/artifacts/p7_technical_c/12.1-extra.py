from tiny_perceptron.multimodal import tone
w=tone(440.0);long=tone(440.0,seconds=.2);lowrate=tone(440.0,sample_rate=8000)
print('duration variant',tuple(long.shape),'same prefix',torch.equal(w,long[:len(w)]))
print('rate variant',tuple(lowrate.shape),'same times',torch.allclose(lowrate,w[::2]))
print('batch',tuple(torch.stack([w,w]).shape))
