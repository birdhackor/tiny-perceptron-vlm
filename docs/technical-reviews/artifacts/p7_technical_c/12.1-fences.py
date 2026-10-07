import torch, sys, json
torch.set_num_threads(1)
print('environment',json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu'}))

print('fence 0')
from tiny_perceptron.multimodal import tone

wave = tone(440.0)
print("樣本形狀", tuple(wave.shape))
print("前三個值", [round(v, 3) for v in wave[:3].tolist()])
print("最小最大", round(wave.min().item(), 2), round(wave.max().item(), 2))


# Owner supplied proportional check
from tiny_perceptron.multimodal import tone
w=tone(440.0);long=tone(440.0,seconds=.2);lowrate=tone(440.0,sample_rate=8000)
print('duration variant',tuple(long.shape),'same prefix',torch.equal(w,long[:len(w)]))
print('rate variant',tuple(lowrate.shape),'same times',torch.allclose(lowrate,w[::2]))
print('batch',tuple(torch.stack([w,w]).shape))

